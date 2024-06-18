import os
import time
import pickle
import torch
import argparse
import numpy as np
from tqdm import tqdm
from model import SASRec
from utils import *
import sys

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def evaluate_all(model, mode, dataset, epoch, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    if mode == 'valid':
        print("validating...")
    if mode == 'test':
        print("testing...")
    # need path to pickle file of list(ranked list)
    users = range(1, usernum + 1)[:1000]
    neg_samples = {}
    if args.sample_type == 'random':
        with open(f"../../../data/{args.dataset}/negative_samples.txt", "r") as f:
            for line in f:
                userid, itemids = line.strip().split(' ', 1)
                userid = int(userid)
                itemids = itemids.split(' ')
                itemids = [int(itemid) for itemid in itemids]
                neg_samples[userid] = itemids[:args.test_neg_num]
    elif args.sample_type == 'pop':
        with open(f"../../../data/{args.dataset}/negative_samples_pop.txt", "r") as f:
            for line in f:
                userid, itemids = line.strip().split(' ', 1)
                userid = int(userid)
                itemids = itemids.split(' ')
                itemids = [int(itemid) for itemid in itemids]
                neg_samples[userid] = itemids[:args.test_neg_num]
    
    ranked_list = []
    answers = []

    batch_u = []
    batch_seq = []
    batch_candidate_idx = []
    all_user_embeds = []
    current_batch_size = 0
    for u in tqdm(users, desc="inferencing..."):
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if mode == 'test':
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        if mode == 'valid':
            target_id = valid[u][0]
        else:
            target_id = test[u][0]
        candidate_idx = neg_samples[u] + [target_id]
        batch_u.append(u)
        batch_seq.append(seq)
        batch_candidate_idx.append(candidate_idx)
        current_batch_size += 1
        if current_batch_size == args.batch_size*2 or u == users[-1]:
            predictions = model.predict(np.array(batch_u), np.array(batch_seq), np.array(batch_candidate_idx))
            user_embeds = model.calc_user_emb(np.array(batch_seq))
            for i in range(current_batch_size):
                candidate_idx = batch_candidate_idx[i]
                prediction = predictions[i]
                
                item_scores = [[candidate_idx[i], prediction[i].item()] for i in range(args.test_neg_num + 1)]
                item_scores = sorted(item_scores, key=lambda x:x[1], reverse=True)
                ranked_items = [x[0] for x in item_scores]
                ranked_list.append(ranked_items)
                answers.append([candidate_idx[-1]])

            batch_u = []
            batch_seq = []
            batch_candidate_idx = []
            current_batch_size = 0
            all_user_embeds.append(user_embeds.cpu().detach())

    with open(f"../../../data/{args.dataset}/ranked_list_{args.sample_type}.txt", "w") as fw:
        for user, ranked_items in zip(users, ranked_list):
            fw.write(f"{user} {' '.join([str(item) for item in ranked_items])}\n")

    all_user_embeds = torch.concat(all_user_embeds, dim=0).cpu().detach().numpy() # [user_num, embed_dim]
    with open(f"../../../data/{args.dataset}/user_embeds_{args.sample_type}.pkl", "wb") as fw:
        pickle.dump(all_user_embeds, fw)
    
    eval_result, eval_str = evaluation(ranked_list, answers)
    print(f'eval result: {eval_str}')
    sys.stdout.flush()

def evaluation(rankings, ground_truth):
    """
    probability: [batch_size, candidate_num]
    ground_truth: [batch_size]
    """
    metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
            + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
            + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
            + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    result_dict = {}
    for metric in metrics:
        result_dict[metric] = []
    # rankings = torch.argsort(rankings, descending=True)  # [batch_size, candidate_num]
    for ranking_list, click_docs in zip(rankings, ground_truth):
        click_docs = set(click_docs)
        ranking_list = ranking_list
        # print(f'click: {click_docs}, ranking: {ranking_list[:10]}')
        for metric in metrics:
            k = int(metric.split('@')[-1])
            if metric.startswith('ndcg@'):
                score, norm = 0.0, 0.0
                for rank, item in enumerate(ranking_list[:k]):
                    if item in click_docs:
                        score += 1.0 / np.log2(rank + 2)
                for rank in range(len(click_docs)):
                    norm += 1.0 / np.log2(rank + 2)
                res = score / max(0.3, norm)
                result_dict[metric].append(res)

            if metric.startswith('recall@'):
                result_dict[metric].append(0)
                for rank, item in enumerate(ranking_list[:k]):
                    if item in click_docs:
                        result_dict[metric][-1] = 1
                        break
            
            if metric.startswith('hit@'):
                result_dict[metric].append(0)
                for rank, item in enumerate(ranking_list[:k]):
                    if item in click_docs:
                        result_dict[metric][-1] += 1

            if metric.startswith('precision@'):
                score, hits_num = 0, 0
                for rank, item in enumerate(ranking_list[:k]):
                    if item in click_docs and item not in ranking_list[:rank]:
                        hits_num += 1
                        score += hits_num / (rank + 1.0)
                result_dict[metric].append(score / max(1.0, len(click_docs)))
    
    result_str = ''
    for metric in metrics:
        result_str += f'{metric}: {round(np.mean(result_dict[metric]), 4)}, '
    return result_dict, result_str

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--test_neg_num', default=100, type=int)
parser.add_argument('--sample_type', default='random', type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir + '_' + args.sample_type):
    os.makedirs(args.dataset + '_' + args.train_dir + '_' + args.sample_type)
with open(os.path.join(args.dataset + '_' + args.train_dir + '_' + args.sample_type, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3,sample_type=args.sample_type)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        print(f'Evaluating and Testing...')
        # evaluate_all(model, "valid", dataset, args)
        evaluate_all(model, "test", dataset, 0, args)
        all_item_embeds = model.calc_item_emb().cpu().detach().numpy()
        with open(f"../../../data/{args.dataset}/item_embeds_{args.sample_type}.pkl", "wb") as fw:
            pickle.dump(all_item_embeds, fw)

        f.close()
        sampler.close()
        print("Done")
        exit(0)

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        pbar = tqdm(total=num_batch, ncols=80, leave=True, unit='b')
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            pbar.set_description("|epoch:{}|loss:{:.3f}|".format(epoch, loss.item())) # expected 0.4~0.6 after init few epochs
            pbar.update(1)
        pbar.close()
    
        if epoch % 10 == 0:
            model.eval()
            print(f'{epoch} Evaluating and Testing...')
            # evaluate_all(model, "valid", dataset, args)
            folder = args.dataset + '_' + args.train_dir + '_' + args.sample_type
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
            evaluate_all(model, "test", dataset, epoch, args)
            all_item_embeds = model.calc_item_emb().cpu().detach().numpy()
            with open(f"../../../data/{args.dataset}/item_embeds_{args.sample_type}.pkl", "wb") as fw:
                pickle.dump(all_item_embeds, fw)
            model.train()
    
    f.close()
    sampler.close()
    print("Done")

    all_user_embeds = pickle.load(open(f"../../../data/{args.dataset}/user_embeds_{args.sample_type}.pkl", "rb"))
    all_item_embeds = pickle.load(open(f"../../../data/{args.dataset}/item_embeds_{args.sample_type}.pkl", "rb"))
    matching_scores = np.matmul(all_user_embeds, all_item_embeds.T) # (n_users, n_items)
    with open(f"../../../data/{args.dataset}/retrieval_list_{args.sample_type}.txt", "w") as fw:
        for user_id, user in enumerate(matching_scores):
            sorted_items = np.argsort(user)[::-1]
            filtered_items = []
            for item_id in sorted_items[:101]:
                if item_id == 0: continue
                filtered_items.append(item_id)
            fw.write(f"{user_id+1} " + " ".join([str(item_id) for item_id in filtered_items]) + "\n")
    embeds = [all_user_embeds, all_item_embeds]
    with open(f"../../../data/{args.dataset}/SASRec_embeddings_{args.sample_type}.pkl", "wb") as fw:
        pickle.dump(embeds, fw)