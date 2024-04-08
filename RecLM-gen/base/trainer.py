# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import os
from collections import deque
import torch.nn.functional as F
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from einops import rearrange
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup, LogitsProcessorList, StoppingCriteriaList, MinLengthLogitsProcessor, \
    TopKLogitsWarper, MaxLengthCriteria, TemperatureLogitsWarper
from base.dataset import BaseDataset
from rl.reward import RewardModel
from utils.tools import masked_mean, whiten, eval_decorator, shift, log_prob, Memory, sync_dict
from base.model import BaseModel
from param import Config


# trainer
class BaseTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=(self.args.train_stage == 'RL'))]     # need for RL
        )
        # Use CUDA_VISIBLE_DEVICES=x to select gpu, do not set the --gpu command param
        self.args.gpu = self.accelerator.device.__str__()
        if self.accelerator.is_main_process:
            print(Config(**vars(args)))
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            if args.train_stage in ['SFT', 'RL']:
                with open(os.path.join(args.output_path, 'config.json'), 'w') as f:
                    json.dump(vars(args), f, indent=4)

        self.actor_critic = BaseModel(args=self.args, device=self.args.gpu)
        if self.accelerator.is_main_process:
            self.actor_critic.print_trainable_parameters()

        self.warped_actor_critic = None
        self.train_data = None
        self.val_data = None
        self.train_loader = None
        self.val_loader = None
        self.lr_scheduler = None
        self.optim = None

        if self.args.train_stage in ['SFT']:
            self.sft_loss_fct = CrossEntropyLoss(reduction='none')
        if self.args.train_stage in ['RL']:
            self.sample_batch = 0
            self.backward_step = 0
            self.logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(15, eos_token_id=self.tokenizer.eos_token_id)])
            self.logits_warper = LogitsProcessorList([TopKLogitsWarper(20), TemperatureLogitsWarper(0.7)])
            self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)])
            self.reward_model = RewardModel(self.args, self.tokenizer)
            self.memories = deque([])

    def prepare(self, train_loader, val_loader):
        named_params = {}
        named_params.update(self.actor_critic.actor_named_parameters)
        if self.args.train_stage in ['RL']:
            named_params.update(self.actor_critic.critic_named_parameters)
        self.optim, self.lr_scheduler = self.get_optimizer_scheduler(named_params, len(self.train_loader))

        self.warped_actor_critic, self.optim, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.actor_critic, self.optim, train_loader, val_loader, self.lr_scheduler
        )

    def get_optimizer_scheduler(self, named_params, batch_per_epoch=None, group_wd_params=True):
        params = [p for n, p in named_params.items() if p.requires_grad]
        assert self.args.weight_decay >= 0
        if group_wd_params and self.args.weight_decay > 0:
            params = [
                {'params': [p for p in params if p.ndim >= 2]},
                {'params': [p for p in params if p.ndim < 2], 'weight_decay': 0},
            ]
        optim = AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay,
                      betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_eps)
        if self.args.train_stage == 'RL':
            scheduler = get_polynomial_decay_schedule_with_warmup(optim, 50*self.accelerator.num_processes, batch_per_epoch*self.args.num_episodes, power=2.0)
        else:
            step_total = batch_per_epoch * (self.args.epoch - self.start_epoch) // self.args.gradient_accumulation_steps
            warmup_iters = int(step_total * self.args.warmup_ratio)
            # scheduler = get_linear_schedule_with_warmup(SFT_optim, warmup_iters, step_total)
            scheduler = get_polynomial_decay_schedule_with_warmup(optim, warmup_iters, step_total, power=2.0)

        return optim, scheduler

    @property
    def device(self):
        return self.actor_critic.device

    @property
    def tokenizer(self):
        return self.actor_critic.tokenizer

    @property
    def model(self):
        if self.args.distributed:
            return self.accelerator.unwrap_model(self.actor_critic)
        return self.actor_critic

    def SFT_Loss(self, logit, label):
        shift_logit = logit[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        loss = self.sft_loss_fct(shift_logit.view(-1, self.actor_critic.model_config.vocab_size), shift_label.view(-1))
        loss = loss.view(label.shape[0], -1)
        # loss = loss.sum(dim=1) / (shift_label != -100).sum(dim=1)  # [bs]
        loss = masked_mean(loss, shift_label != -100, dim=1)  # [bs]
        return loss

    def SFT_train_batch(self, batch):
        """ SFT training for a batch with 'bs' samples. The batch dict is got by collate_fn.
        :param batch: A dict includes "complete_text_data" and "complete_label_ids";
                        batch[“complete_text_data”] is a dict includes "input_ids"->[bs, seq_len] and "attention_mask"->[bs, seq_len]
                        batch["complete_label_ids"]->[bs, seq_len] is label of SFT.
        :return: Return the batch dict with "loss"->[bs], which is the loss of each train sample.
        """
        with self.accelerator.accumulate(self.warped_actor_critic):
            # print(f'parameter: ', list(self.actor_critic.actor_named_parameters.values())[0].data.abs().max())
            # self.accelerator.wait_for_everyone()
            input_data, labels = batch['complete_text_data'], batch['complete_label_ids']
            results = self.warped_actor_critic.forward(scope=self.actor_critic.actor_lora_scope, **input_data)
            loss = self.SFT_Loss(results.logits, labels)

            self.accelerator.backward(loss.mean())  # auto divide accumulate step, sync grad if arrive accumulate step
            # print(f'grad: ', list(self.actor_critic.actor_named_parameters.values())[0].grad.abs().max())
            # self.accelerator.wait_for_everyone()

            if self.accelerator.sync_gradients:
                # print(f'sync grad: ', list(self.actor_critic.actor_named_parameters.values())[0].grad.abs().max())
                # self.accelerator.wait_for_everyone()
                if self.args.clip_grad_norm > 0:
                    total_norm = self.accelerator.clip_grad_norm_(self.optim.param_groups[0]['params'], self.args.clip_grad_norm)
                    # writer.add_scalars('training/total_norm', {f'epoch{epoch}': float(total_norm)}, step_i)

            self.optim.step()
            self.lr_scheduler.step()
            self.optim.zero_grad()

        batch['loss'] = loss
        return batch

    def compute_adv(self, old_action_values, rewards, action_mask):
        # reference to implementation of trl lib: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1132
        if self.args.whiten_reward:
            whitened_rewards = whiten(rewards, action_mask, shift_mean=False, dim=None)
        else:
            whitened_rewards = rewards
        last_gae_lam = 0
        advantages_reversed = []
        gen_length = torch.sum(action_mask, dim=1).max().item()
        for time_step in range(1, gen_length + 1):
            next_values = old_action_values[:, -time_step + 1] if time_step > 1 else 0.0
            delta = whitened_rewards[:, -time_step] + self.args.gamma * next_values - old_action_values[:, -time_step]
            last_gae_lam = delta + self.args.gamma * self.args.lam * last_gae_lam
            advantages_reversed.append(last_gae_lam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        advantages = F.pad(advantages, (whitened_rewards.size(1) - gen_length, 0), value=0.0)
        returns = (advantages + old_action_values) * action_mask
        whitened_advantages = whiten(advantages, action_mask, dim=None).detach() * action_mask
        return whitened_advantages, returns

    @torch.no_grad()
    @eval_decorator
    def RL_sample_batch(self, batch):
        """ Sampling the responses of input batch with 'bs' samples for RL. The batch is got by collate_fn.
        :param batch: A dict includes "input_data";
                        batch["input_data"] is a dict includes "input_ids"->[bs, seq_len] and "attention_mask"->[bs, seq_len]
        :return: Return output_text is the 'list[str]' with the outputs of each sample. the size is 'sample_num*bs'.
                 Such as sample_num = 3, output_text[0] is the greedy search result of the 0-th sample.
                 output_text[1*bs+0] and output_text[2*bs+0] is the sample result of the 0-th sample.
        """
        input_data = batch['input_data']
        input_ids_length = input_data['input_ids'].shape[1]
        output_ids = self.actor_critic.actor_model.greedy_search(
            **input_data, logits_processor=self.logits_processor, stopping_criteria=self.stopping_criteria
        )
        output_text = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
        if self.args.sample_num > 1 and self.args.lr > 0:
            sample_input_data = {
                'input_ids': input_data['input_ids'].repeat((self.args.sample_num-1, 1)).contiguous(),
                'attention_mask': input_data['attention_mask'].repeat((self.args.sample_num-1, 1)).contiguous()
            }
            output_ids = self.actor_critic.actor_model.sample(
                **sample_input_data, logits_processor=self.logits_processor, logits_warper=self.logits_warper, stopping_criteria=self.stopping_criteria
            )
            output_text += self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
        # output_dict = [{'index': j, 'output_text': output_text[i*bs+j]} for i in range(self.args.sample_num) for j in range(bs)]
        return output_text

    @torch.no_grad()
    @eval_decorator
    def PPO_process_batch(self, complete_data, action_mask, total_reward):
        """ This function is used to prepare data in 'self.memories' for PPO training.
        :param complete_data: is a dict with 'input_ids'->[sample_num*bs, seq_len] and 'attention_mask'->[sample_num*bs, seq_len]. Generated by the input and sampled output.
        :param action_mask: Same shape with complete_data['input_ids']. The places of output tokens are set True. Others are False.
        :param total_reward: Calculated by reward model. Same shape with complete_data['input_ids'], indicating the reward of each token.
        :return: None.
        """
        if self.args.lr <= 0:
            return
        for i in range(0, total_reward.shape[0], 16):
            temp_learn_data = {
                'complete_data': {
                    'input_ids': complete_data['input_ids'][i: i + 16],
                    'attention_mask': complete_data['attention_mask'][i: i + 16],
                },
                'action_mask': action_mask[i: i + 16],
                'total_reward': total_reward[i: i + 16],
            }
            mini_sample_num = temp_learn_data['total_reward'].shape[0]
            old_action_values = self.actor_critic.forward(scope=self.actor_critic.critic_lora_scope, **temp_learn_data['complete_data'])

            old_sequence_logit = self.actor_critic.forward(scope=self.actor_critic.actor_lora_scope, **temp_learn_data['complete_data']).logits
            old_sequence_dists = torch.softmax(old_sequence_logit, dim=-1)
            old_sequence_dists_shifted = shift(old_sequence_dists, shift=1, dim=-2).contiguous()
            old_sequence_log_probs_shifted = log_prob(old_sequence_dists_shifted, temp_learn_data['complete_data']['input_ids'])

            ref_sequence_logit = self.actor_critic.forward(scope='base', **temp_learn_data['complete_data']).logits
            ref_sequence_dists = torch.softmax(ref_sequence_logit, dim=-1)
            ref_sequence_dists_shifted = shift(ref_sequence_dists, shift=1, dim=-2).contiguous()
            ref_sequence_log_probs_shifted = log_prob(ref_sequence_dists_shifted, temp_learn_data['complete_data']['input_ids'])

            kl_penalty = (old_sequence_log_probs_shifted - ref_sequence_log_probs_shifted) * action_mask  # 其他kl
            rewards_penalized = temp_learn_data['total_reward'] - kl_penalty * self.args.kl_coef
            whitened_advantages, returns = self.compute_adv(old_action_values, rewards_penalized, action_mask)

            max_len = 1024
            for idx in range(mini_sample_num):
                ignore_index = torch.not_equal(complete_data['input_ids'][idx, ...], self.tokenizer.pad_token_id)
                # if ignore_index.sum() > max_len:
                #     continue
                (
                    sequence_i,
                    action_mask_i,
                    old_action_values_i,
                    old_sequence_log_probs_shifted_i,
                    ref_sequence_log_probs_shifted_i,
                    whitened_advantages_i,
                    returns_i
                ) = (
                    rearrange(complete_data['input_ids'][idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous(),
                    rearrange(action_mask[idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous(),
                    rearrange(old_action_values[idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous(),
                    rearrange(old_sequence_log_probs_shifted[idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous(),
                    rearrange(ref_sequence_log_probs_shifted[idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous(),
                    rearrange(whitened_advantages[idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous(),
                    rearrange(returns[idx, ...][ignore_index][-max_len:], '... -> 1 ...').contiguous()
                )

                # append train sample
                self.memories.append(
                    Memory(
                        sequence_i,
                        action_mask_i,
                        old_action_values_i,
                        old_sequence_log_probs_shifted_i,
                        ref_sequence_log_probs_shifted_i,
                        whitened_advantages_i,
                        returns_i
                    )
                )

    def learn_PPO(self):
        """ This function does RL training by the data in 'self.memories'.
        :return: ppo_stat, which has the metrics of PPO learning.
        """
        self.actor_critic.train()
        ppo_stat = {
            'training/backward_step': 0.0,
            'training/actor_loss_mean': 0.0,
            'training/critic_loss_mean': 0.0,
            'training/entropy_loss_mean': 0.0,
            'training/total_loss_mean': 0.0,
            'training/approx_kl': 0.0,
            'training/policy_kl': 0.0,
            'training/action_clip_frac': 0.0,
            'training/critic_clip_frac': 0.0,
        }
        ratio_clip_range = 0.2
        value_clip_range = 0.2
        for epoch in range(self.args.epoch):
            policy_kl = float('inf')
            for (
                    _sequence,
                    _action_mask,
                    _old_action_values,
                    _old_sequence_log_probs_shifted,
                    _ref_sequence_log_probs_shifted,
                    _whitened_advantages,
                    _returns
            ) in self.memories:
                self.backward_step += 1
                with self.accelerator.accumulate(self.warped_actor_critic):
                    # print(f'parameter: ', list(self.actor_critic.actor_named_parameters.values())[0].data.abs().max())
                    # self.accelerator.wait_for_everyone()
                    # print(f'parameter: ', list(self.actor_critic.critic_named_parameters.values())[0].data.abs().max())
                    # self.accelerator.wait_for_everyone()
                    sequence_logit = self.warped_actor_critic.forward(self.actor_critic.actor_lora_scope, _sequence, attention_mask=_sequence != 0).logits
                    sequence_dists = torch.distributions.categorical.Categorical(logits=sequence_logit)
                    sequence_dists_shifted = shift(sequence_dists.probs, shift=1, dim=-2).contiguous()
                    sequence_log_probs_shifted = log_prob(sequence_dists_shifted, _sequence)
                    # entropy loss
                    entropy_losses = masked_mean(sequence_dists.entropy(), _action_mask)
                    # action loss
                    log_ratio = sequence_log_probs_shifted - _old_sequence_log_probs_shifted
                    ratio = torch.exp(log_ratio)
                    action_losses1 = -_whitened_advantages * ratio
                    action_losses2 = -_whitened_advantages * torch.clamp(ratio,
                                                                         min=1.0 - ratio_clip_range,
                                                                         max=1.0 + ratio_clip_range)
                    action_losses = masked_mean(torch.max(action_losses1, action_losses2), _action_mask)
                    # actor loss
                    actor_loss = action_losses - self.args.entropy_weight * entropy_losses
                    self.accelerator.backward(actor_loss)
                    # critic loss
                    action_values = self.warped_actor_critic.forward(self.actor_critic.critic_lora_scope, _sequence, attention_mask=_sequence != 0)
                    new_values_clipped = torch.clamp(action_values,
                                                     min=_old_action_values - value_clip_range,
                                                     max=_old_action_values + value_clip_range)
                    critic_losses1 = torch.square(action_values - _returns)
                    critic_losses2 = torch.square(new_values_clipped - _returns)
                    critic_losses = 0.5 * masked_mean(torch.max(critic_losses1, critic_losses2), _action_mask)
                    self.accelerator.backward(self.args.vf_coef * critic_losses)
                    # print(f'grad: ', list(self.actor_critic.actor_named_parameters.values())[0].grad.abs().max())
                    # self.accelerator.wait_for_everyone()
                    # print(f'grad: ', list(self.actor_critic.critic_named_parameters.values())[0].grad.abs().max())
                    # self.accelerator.wait_for_everyone()

                    if self.accelerator.sync_gradients:
                        # print(f'sync grad: ', list(self.actor_critic.actor_named_parameters.values())[0].grad.abs().max())
                        # self.accelerator.wait_for_everyone()
                        # print(f'sync grad: ', list(self.actor_critic.critic_named_parameters.values())[0].grad.abs().max())
                        # self.accelerator.wait_for_everyone()
                        if self.args.clip_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(self.optim.param_groups[0]['params'], self.args.clip_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

                approx_kl = 0.5 * masked_mean(log_ratio ** 2, _action_mask)
                policy_kl = masked_mean(-log_ratio, _action_mask)
                critic_clip_frac = masked_mean(torch.gt(critic_losses2, critic_losses1).float(), _action_mask)
                action_clip_frac = masked_mean(torch.gt(action_losses2, action_losses1).float(), _action_mask)
                total_losses = action_losses + self.args.vf_coef * critic_losses - self.args.entropy_weight * entropy_losses
                ppo_stat['training/backward_step'] += 1
                ppo_stat['training/actor_loss_mean'] += float(action_losses)
                ppo_stat['training/critic_loss_mean'] += float(critic_losses)
                ppo_stat['training/entropy_loss_mean'] += float(entropy_losses)
                ppo_stat['training/total_loss_mean'] += float(total_losses)
                ppo_stat['training/approx_kl'] += float(approx_kl)
                ppo_stat['training/policy_kl'] += float(policy_kl)
                ppo_stat['training/action_clip_frac'] += float(action_clip_frac)
                ppo_stat['training/critic_clip_frac'] += float(critic_clip_frac)

            policy_kl = self.accelerator.reduce(policy_kl, reduction='mean')
            if self.accelerator.is_main_process:
                print(f"Batch: {self.sample_batch} | Epoch: {epoch} | backward_step: {self.backward_step} | "
                      f"RL_lr: {self.lr_scheduler.get_lr()[0]:.7f} | policy_kl: {policy_kl:.6f}")
            if policy_kl > self.args.policy_kl_threshold:
                break
            np.random.shuffle(self.memories)

        self.accelerator.wait_for_everyone()
        return ppo_stat

    def Adapter_merge(self):
        if self.args.train_stage == 'SFT_Merge':
            train_epoch = self.actor_critic.load_parameters(self.args.SFT_load)
            model = self.actor_critic.lora_model.merge_and_unload(progressbar=True)
            save_path = os.path.join(self.args.output_path, f'SFT_Epoch{train_epoch:02d}')
            model.save_pretrained(save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(save_path)
        elif self.args.train_stage == 'RL_Merge':
            train_step = self.actor_critic.load_parameters(self.args.RL_load)
            self.actor_critic.lora_model.delete_adapter(self.actor_critic.critic_lora_scope)
            model = self.actor_critic.lora_model.merge_and_unload(progressbar=True)
            save_path = os.path.join(self.args.output_path, f'RL_Step{train_step}')
            model.save_pretrained(save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(save_path)
        else:
            raise NotImplementedError

    def dataset_prepare(self):
        self.train_data = BaseDataset(self.args, self.tokenizer, 'train')
        self.val_data = BaseDataset(self.args, self.tokenizer, 'val')
        self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.train_data.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.val_data.collate_fn, drop_last=False)

        self.prepare(self.train_loader, self.val_loader)

    def SFT_train(self):
        best_val_loss = float('inf')
        if self.args.dry:
            best_val_loss = self.SFT_val_loss(0)
        for epoch in range(1, self.args.epoch+1):
            if hasattr(self.train_data, "set_epoch"):
                self.train_data.set_epoch(epoch-1)
            pbar = tqdm(total=len(self.train_loader), ncols=210, disable=not self.accelerator.is_local_main_process)
            self.train()
            for step_i, batch in enumerate(self.train_loader):
                batch = self.SFT_train_batch(batch)

                if self.accelerator.sync_gradients:
                    pbar.update(self.args.gradient_accumulation_steps)
            pbar.close()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.actor_critic.save_parameters(f"Epoch{epoch:02d}")
            if epoch < self.args.val_epoch:
                continue
            val_loss = self.SFT_val_loss(epoch)
            if val_loss < best_val_loss and self.accelerator.is_main_process:
                best_val_loss = val_loss
                self.actor_critic.save_parameters("BEST_EVAL_LOSS")

    def SFT_val_loss(self, epoch: int):
        torch.cuda.empty_cache()
        total_loss = 0.0
        total_count = 0
        pbar = tqdm(total=len(self.val_loader), ncols=200, disable=not self.accelerator.is_main_process)
        for step_i, batch in enumerate(self.val_loader):
            labels = batch['complete_label_ids']
            results = self.actor_critic.forward(scope=self.actor_critic.actor_lora_scope, **batch['complete_text_data'])
            loss = self.SFT_Loss(results.logits, labels).detach()
            total_loss += loss.sum()
            total_count += loss.shape[0]
            pbar.update(1)

        val_loss = total_loss/total_count
        if self.accelerator.is_main_process:
            print(f'Epoch {epoch} | SFT_Val_Loss: {val_loss:.4f}\n')
        return val_loss

    def SFT_val_inference(self, epoch: int):
        pass

    def RL_train(self):
        best_val_reward = -float('inf')
        if self.args.dry and self.args.lr > 0:
            self.RL_val(0)
        for eps in range(self.args.num_episodes):
            if hasattr(self.train_data, "set_epoch"):
                self.train_data.set_epoch(eps)
            pbar = tqdm(total=len(self.train_loader), ncols=150, disable=not self.accelerator.is_main_process)
            for batch in self.train_loader:
                self.accelerator.wait_for_everyone()
                self.sample_batch += 1

                # sampling train data
                output_text = self.RL_sample_batch(batch)

                # get reward from reward model
                output_reward = self.reward_model.get_reward(batch, output_text)
                complete_data, action_mask, total_reward = output_reward.complete_data, output_reward.action_mask, output_reward.total_reward
                # process ppo data

                self.PPO_process_batch(complete_data, action_mask, total_reward)

                pbar.set_description(f'RL learning in episode: {eps} | sample_batch: {self.sample_batch} | example: {len(self.memories)} | '
                                     f'max_length: {complete_data["input_ids"].shape[1]}')
                pbar.update(1)

                if self.sample_batch % self.args.learn_batch == 0 and self.args.lr > 0:
                    # ppo learning
                    self.lr_scheduler.step()
                    ppo_stat = self.learn_PPO()
                    # logging ppo stat
                    ppo_stat = sync_dict(self.accelerator, ppo_stat)
                    # if self.accelerator.is_main_process:
                    #     for k, v in ppo_stat.items():
                    #         self.writer.add_scalar(k, v / ppo_stat['training/backward_step'], self.sample_batch)
                    self.memories.clear()

                if self.sample_batch % self.args.val_save_step == 0:
                    if self.args.lr > 0:
                        if self.accelerator.is_main_process:
                            self.actor_critic.save_parameters(f'{self.sample_batch}step')
                        val_reward = self.RL_val(self.sample_batch)
                        if val_reward > best_val_reward and self.accelerator.is_main_process:
                            best_val_reward = val_reward
                            self.actor_critic.save_parameters("BEST_EVAL_REWARD")

            pbar.close()
        print('RL training complete')

    def RL_val(self, step: int):
        pbar = tqdm(total=len(self.val_loader), ncols=150, disable=not self.accelerator.is_main_process)
        total_reward = 0.0
        total_count = 0
        for step_i, batch in enumerate(self.val_loader):
            self.accelerator.wait_for_everyone()
            input_ids_length = batch['input_data']['input_ids'].shape[1]
            val_model = self.actor_critic.base_model if step == 0 else self.actor_critic.actor_model
            output_ids = val_model.greedy_search(**batch['input_data'], stopping_criteria=self.stopping_criteria)
            output_text = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)

            output_reward = self.reward_model.get_reward(batch, output_text, only_reward=True)
            reward = output_reward.reward
            total_reward += sum(reward)
            total_count += len(reward)
            pbar.update(1)

        val_reward = total_reward/total_count
        print(f'Step {step} | RL_Val_Reward: {val_reward:.4f}\n')
        return val_reward


if __name__ == '__main__':
    pass
