import argparse
import collections
import os
import json
import torch

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from index.datasets import EmbDataset
from index.models.rqvae import RQVAE

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Generate item indices from trained RQ-VAE model")
    parser.add_argument("--dataset", type=str, default="Movies", help="Dataset name")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to RQ-VAE checkpoint. If not provided, uses ./ckpt/{dataset}/best_collision_model.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. If not provided, uses ../data/{dataset}/")
    parser.add_argument("--metas_file", type=str, default=None,
                        help="Path to metas.jsonl file for extracting item_ids. If not provided, uses sequential indices.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser.parse_args()


cmd_args = parse_args()

# Set default paths if not provided
dataset = cmd_args.dataset
ckpt_path = cmd_args.ckpt_path or f"./ckpt/{dataset}/best_collision_model.pth"
output_dir = cmd_args.output_dir or f"../data/{dataset}/"
device = torch.device(cmd_args.device)

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Output file paths
index_output_file = os.path.join(output_dir, f"{dataset}.index.json")
item2id_output_file = os.path.join(output_dir, f"{dataset}.item2id")

print(f"Dataset: {dataset}")
print(f"Checkpoint: {ckpt_path}")
print(f"Output directory: {output_dir}")
print(f"Index output: {index_output_file}")
print(f"Item2id output: {item2id_output_file}")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
train_args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(train_args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=train_args.num_emb_list,
                  e_dim=train_args.e_dim,
                  layers=train_args.layers,
                  dropout_prob=train_args.dropout_prob,
                  bn=train_args.bn,
                  loss_type=train_args.loss_type,
                  quant_loss_weight=train_args.quant_loss_weight,
                  kmeans_init=train_args.kmeans_init,
                  kmeans_iters=train_args.kmeans_iters,
                  sk_epsilons=train_args.sk_epsilons,
                  sk_iters=train_args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data, num_workers=train_args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

# Prepare item_id list
print("\n" + "="*50)
print("Generating item_id mappings...")

item_ids = []
if cmd_args.metas_file and os.path.exists(cmd_args.metas_file):
    # Load item_ids from metas.jsonl
    print(f"Loading item_ids from: {cmd_args.metas_file}")
    with open(cmd_args.metas_file, 'r', encoding='utf-8') as f:
        metas = json.load(f)

    # Sort by emb_idx if available, otherwise use dict order
    if metas and 'emb_idx' in next(iter(metas.values())):
        print("Sorting by 'emb_idx' field")
        sorted_items = sorted(metas.items(), key=lambda x: x[1].get('emb_idx', 0))
        item_ids = [item_id for item_id, _ in sorted_items]
    else:
        print("Using dictionary order (no emb_idx found)")
        item_ids = list(metas.keys())

    if len(item_ids) != len(all_indices):
        print(f"WARNING: Metas has {len(item_ids)} items but embeddings have {len(all_indices)} items")
        print(f"Using first {len(all_indices)} items from metas")
        item_ids = item_ids[:len(all_indices)]
else:
    # Use sequential indices as item_ids
    print(f"No metas file provided, using sequential indices (0, 1, 2, ...)")
    item_ids = [str(i) for i in range(len(all_indices))]

# Generate .index.json (key: index, value: token list)
all_indices_dict = {}
for idx, indices in enumerate(all_indices.tolist()):
    all_indices_dict[str(idx)] = list(indices)

print(f"\nSaving index file: {index_output_file}")
with open(index_output_file, 'w', encoding='utf-8') as fp:
    json.dump(all_indices_dict, fp, indent=2, ensure_ascii=False)

# Generate .item2id file (format: item_id index)
print(f"Saving item2id file: {item2id_output_file}")
with open(item2id_output_file, 'w', encoding='utf-8') as f:
    for idx, item_id in enumerate(item_ids):
        f.write(f"{item_id} {idx}\n")

print("\n" + "="*50)
print("Generation completed successfully!")
print(f"Total items: {len(all_indices)}")
print(f"Index file: {index_output_file}")
print(f"Item2id file: {item2id_output_file}")
print("="*50)