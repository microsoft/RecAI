# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

def ExpCollator(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = torch.stack([f["labels"] for f in features]) if "labels" in features[0] else None

    max_len = input_ids.size(1)
    # reshape user_pos to (batch_size*u_n)
    user_pos = []
    user_ids = []
    item_seq = []
    item_pos = []
    item_ids = []
    for i, f in enumerate(features):
        for x in f['user_pos']:
            user_pos.append(x + i*max_len)
        for x in f['user_ids']:
            user_ids.append(x)
        for x in f['item_seq']:
            item_seq.append(x)
        for x in f['item_pos']:
            item_pos.append(x + i*max_len)
        for x in f['item_ids']:
            item_ids.append(x)

    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "user_pos": torch.tensor(user_pos, dtype=torch.long),
        "user_ids": torch.tensor(user_ids, dtype=torch.long),
        "item_seq": torch.tensor(item_seq, dtype=torch.long),
        "item_pos": torch.tensor(item_pos, dtype=torch.long),
        "item_ids": torch.tensor(item_ids, dtype=torch.long)
    }
    
    if labels is not None:
        return_dict["labels"] = labels

    if "answers" in features[0]:
        return_dict["answers"] = [f["answers"] for f in features]
    
    if 'type' in features[0]:
        return_dict['type'] = [f['type'] for f in features]
    
    return return_dict