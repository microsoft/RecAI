# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import *

import numpy as np
import torch
from llm4crs.buffer import CandidateBuffer
from llm4crs.corups import BaseGallery
from llm4crs.utils import get_topk_index, normalize_np
from loguru import logger
from unirec.model.base.recommender import BaseRecommender
from unirec.utils import general


class RecModelTool:
    def __init__(
        self,
        name: str,
        desc: str,
        model_fpath: str,
        item_corups: BaseGallery,
        buffer: CandidateBuffer,
        rec_num: int = None,
        max_hist_len: int = 50,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.item_corups = item_corups
        self.buffer = buffer
        self.name = name
        self.desc = desc
        model, config = general.load_model_freely(model_fpath, device)
        self.device = torch.device(device)
        self.model: BaseRecommender = model.to(self.device)
        self.rec_num = rec_num
        self.max_hist_len = max_hist_len
        self.temperature = temperature
        self._mode = "accuracy"

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        self._mode = mode

    def run(self, inputs: str):
        info = ""
        try:
            inputs = json.loads(inputs)
        except json.decoder.JSONDecodeError as e:
            info += f"{self.name}: Input format error."
            logger.debug(e)
            self.buffer.track(self.name, inputs, info)
            return info

        schema = inputs.get("schema", "popularity")
        rec_num = self.rec_num
        prefer = inputs.get("prefer", [])
        unwanted = inputs.get("unwanted", [])

        candidates = self.buffer.get()
        info += (
            f"Before {self.name}: There are {len(candidates)} candidates in buffer. "
        )
        if len(candidates) == 0:
            info += "Stop execution. \n"
        else:
            info += "\n"

        try:
            if (len(prefer)) > 0:
                prefer = self.item_corups.fuzzy_match(prefer, "title")
                prefer_game_ids = self.item_corups.convert_title_2_info(
                    prefer, col_names="id"
                )["id"]
                if isinstance(prefer_game_ids, int):
                    prefer_game_ids = [prefer_game_ids]
            else:
                prefer_game_ids = []

            if (len(unwanted)) > 0:
                unwanted = self.item_corups.fuzzy_match(unwanted, "title")
                unwanted_game_ids = self.item_corups.convert_title_2_info(
                    unwanted, col_names="id"
                )["id"]
                if isinstance(unwanted_game_ids, int):
                    unwanted_game_ids = [unwanted_game_ids]
            else:
                unwanted_game_ids = []

            prof_ids = torch.tensor(
                prefer_game_ids, dtype=torch.long, device=self.device
            )
            prof_len = torch.tensor(
                [len(prof_ids)],
                dtype=torch.int,
            )

            if schema not in {"popularity", "preference", "similarity"}:
                info += f"{self.name}: ranking schema switch to 'popularity' because only ['popularity', 'preference', 'similarity'] are supported but get {schema}. \n"
                schema = "popularity"

            similarity_score = self.buffer.similarity
            if schema == "similarity" and (similarity_score is None):
                schema = "popularity"
                info += f"{self.name}: ranking schema switch to 'popularity' because similarity scores have not been calculated with Similarity Filtering Tool. \n"

            if rec_num is None:
                N = len(candidates)
            else:
                N = min(len(candidates), rec_num)

            if len(prefer_game_ids) > 0:
                schema = "preference"
                logger.debug(
                    "'prefer' info is given, change the ranking schema to 'preference'."
                )
                info += f"{self.name}: ranking schema switch to 'preference' because 'prefer' info is given. \n"

            if schema == "preference" and (len(prefer_game_ids) <= 0):
                schema = "popularity"
                logger.debug(
                    "'prefer' info is not given, change the ranking schema to 'popularity'."
                )
                info += f"{self.name}: ranking schema switch to 'popularity' because 'prefer' info is not given. \n"

            if schema == "popularity":
                # rank by popularity
                rec_id = self.rank_by_pop(candidates, N, unwanted_game_ids)
                self.buffer.push(self.name, rec_id)
                info += f"{self.name}: Items are ranked according to their popularities. The ranked game ids are stored in buffer and are visible to all tools. \n"

            elif (schema == "similarity") and (similarity_score is not None):
                # rank by similarity
                rec_id = self.rank_by_sim(
                    candidates, N, similarity_score, unwanted_game_ids
                )
                self.buffer.push(self.name, rec_id)
                info += f"{self.name}: Items are ranked according to the similarity with seed items. The ranked game ids are stored in buffer and are visible to all tools. \n"

            else:
                # rank by preference
                candidates = torch.tensor(
                    candidates, dtype=torch.long, device=self.device
                )
                model_input = {
                    "item_seq": prof_ids.view(1, -1).to(self.device),
                    "item_seq_len": prof_len.to(self.device),
                    "item_id": candidates,
                }

                scores = self.model.predict(model_input)
                scores = torch.from_numpy(scores).squeeze(0)

                scores = torch.sigmoid(scores)

                if scores.dim() < 1:
                    scores = scores.view(1, -1)

                unwanted_game_ids = prefer_game_ids + unwanted_game_ids
                if len(unwanted_game_ids) > 0:
                    unwanted_game_ids = torch.tensor(
                        unwanted_game_ids, dtype=torch.long, device=self.device
                    )
                    mask = torch.tensor(
                        [x in unwanted_game_ids for x in candidates], device=self.device
                    )
                    scores[mask] = 1e-15

                if self._mode == "diversity":
                    _idx = torch.multinomial(scores, num_samples=N)
                else:
                    _, _idx = torch.topk(scores, k=N)

                rec_id = torch.gather(candidates, -1, _idx.view(-1)).cpu().numpy()
                self.buffer.push(self.name, rec_id)
                # return f"Here are candidates id ranked according to human's profile: [{', '.join(map(str, rec_id))}]."
                info += "{self.name}: Items are ranked according to the preference. The ranked game ids are stored in buffer and are visible to all tools. \n"

        except Exception as e:
            info += f"{self.name}: some thing went wrong in execution, the tool is broken for current input."

        self.buffer.track(self.name, inputs, info)

        return info

    def rank_by_pop(
        self, candidates: List[int], N: int, masked_items: List[int]
    ) -> List[int]:
        pop = self.item_corups.convert_id_2_info(candidates, col_names=["visited_num"])[
            "visited_num"
        ]
        return self._rank_by_x(candidates, N, np.log(np.array(pop) + 1), masked_items)

    def rank_by_sim(
        self,
        candidates: List[int],
        N: int,
        sim_scores: List[float],
        masked_items: List[int],
    ) -> List[int]:
        return self._rank_by_x(candidates, N, sim_scores, masked_items)

    def _rank_by_x(
        self,
        candidates: List[int],
        N: int,
        scores: List[float],
        masked_items: List[int],
    ) -> List[int]:
        N = min(len(candidates), N)
        candidates = np.array(candidates)
        scores = np.array(scores)
        if self._mode == "diversity":
            scores[np.isin(candidates, np.array(masked_items))] = 0.0
            prob = normalize_np(scores)
            items = np.random.choice(candidates, size=N, p=prob, replace=False)
        else:
            scores[np.isin(candidates, np.array(masked_items))] = -100000.0
            _id = get_topk_index(scores[None, :], topk=N)[0]
            items = candidates[_id]
        return items.tolist()
