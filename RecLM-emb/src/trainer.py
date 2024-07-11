"""
The following code is modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/trainer.py
"""

from transformers.trainer import *

class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model
        
        model = model.model #### because the BertModel is wrapped in BiEncoderModel

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            # If the model is on the GPU, it still works!
            if self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                model, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapters = model.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)