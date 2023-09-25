# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from typing import *
import numpy as np
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

class SentBERTEngine:
    def __init__(self, corpus: np.ndarray, id: np.ndarray, case_sensitive: bool=False, model_name: str='all-mpnet-base-v2'):
        self.corpus = corpus
        self.id = id
        if not case_sensitive:
            corpus = [doc.lower() for doc in corpus]
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        embeddings = self.model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
        self.engine = IndexFlatIP(embeddings.size(1))
        self.engine.add(embeddings.cpu())
        self.case_sensitive = case_sensitive

    
    def __call__(self, query:Union[str, List[str]], return_doc=False, topk: int=None, thres: float=None) ->  np.ndarray:
        if not self.case_sensitive:
            query = query.lower() if isinstance(query, str) else [q.lower() for q in query]
        q_emb = self.model.encode(query, normalize_embeddings=True, convert_to_tensor=True).cpu()
        if q_emb.dim() == 1:
            q_emb = q_emb.view(1, -1)
        score, idx = self.engine.search(q_emb, topk)
        # if len(score.shape) > 1 and isinstance(query, str) and score.shape[0] == 1:
        #     score, idx = score.squeeze(0), idx.squeeze(0)
        if thres is not None:
            idx = idx[score >= thres]
        if return_doc:
            res = self.corpus[idx]
        else:
            res = self.id[idx]

        if isinstance(query, str):
            res = np.squeeze(res, axis=0)
        return res