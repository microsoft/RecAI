# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import *
import numpy as np
from numpy.typing import NDArray

from llm4crs.corups import BaseGallery
from llm4crs.prompt import *


class CandidateBuffer:

    def __init__(self, item_corups: BaseGallery, num_limit: int=None) -> None:
        self.item_corups = item_corups
        self.init_memory = []   # to store user given candidates, deprecated
        self.memory = list(range(1, len(self.item_corups)))
        self.plans = []
        self.failed_plans = []
        self.num_limit = num_limit
        self.tracker = []
        self.similarity = None

    
    def save_similarity(self, sim):
        if self.num_limit is not None:
            self.similarity = sim[:self.num_limit]
        else:
            self.similarity = sim


    def track(self, tool, input: str=None, output: str=None):
        self.tracker.append({'tool': tool, 'input': input, 'output': output})

    @property
    def track_info(self):
        res = "\n".join(["{num}: {tool} (input: {input}, output: {output}); ".format(num=i, **t) for i, t in enumerate(self.tracker)])
        return res

    
    def clear_tracks(self):
        self.tracker = []


    def init_candidates(self, inputs: str) -> str:
        """ Init init_memory with user given candidates
        
        Args:
            inputs (str): candidate names written to the buffer, names are concantenated by ',' from LLM
        """
        if len(self.init_memory) != 0:
            print("Warning: `init_candidates` should be called when the buffer is empty")
        try:
            candidates = [x.strip() for x in inputs.split(';;')]
            candidates = self.item_corups.fuzzy_match(candidates, 'title')
            candidate_ids = self.item_corups.convert_title_2_info(candidates, col_names='id')['id']
            if self.num_limit is not None:
                self.init_memory = list(candidate_ids)[:self.num_limit]
                self.memory = list(candidate_ids)[:self.num_limit]
            else:
                self.init_memory = list(candidate_ids)
                self.memory = list(candidate_ids)
            output = f"{len(candidate_ids)} candidate games are given by human in conversation and stored in buffer. " \
                      "Those games are visible to other tools to be further filtered and ranked."
        except Exception as e:
            print(e)
            output = "Storing games failed, please retry."

        self.track(TOOL_NAMES["BufferStoreTool"], inputs, output)
        return output



    def current_candidates(self) -> List[int]:
        """ Return current candidate game ids 
        
        Returns:
            candidates (List[int]): current candidate game ids 
        """
        # if len(self.memory) <= 0:
        #     return self.init_memory
        # else:
        return self.memory


    def update(self, tool: str, candidates: Union[List[int], NDArray[np.int32]]) -> None:
        """ Update candidates buffer memory

        Args:
            tool (str): tool name where the update is triggered 
            candidates (List[int] or np.ndarray[int]): candidates written to the buffer
        """

        # type check
        assert isinstance(tool, str), f"`tool` should be string, bet got {type(tool)}"
        assert (isinstance(candidates, list) or isinstance(candidates, np.ndarray)), \
            "`candidates` should be a list or numpy.ndarray"
        if len(candidates) > 0:
            assert isinstance(candidates[0], int) or isinstance(candidates[0], np.integer), f"`candidates` should be list of int but got {type(candidates[0])}"
            if self.num_limit is not None:
                self.memory = list(candidates)[:self.num_limit]
            else:
                self.memory = list(candidates)
        else:
            self.memory = []
        
        self.plans.append((tool, len(candidates)))


    def get(self) -> List[int]:
        """ Get the candidates in the buffer. """
        return self.current_candidates()
    

    def push(self, tool: str, candidates: Union[List[int], NDArray[np.int32]]) -> None:
        self.update(tool, candidates)


    def clear(self) -> None:
        """Clear all state of the buffer.
        
        The method would be called when one turn chat is finished
        """
        self.init_memory = []
        self.memory = list(range(1, len(self.item_corups)))
        self.plans = []
        self.failed_plans = []
        self.similarity = None



    def replan(self, inputs: str) -> str:
        """Clear the buffer for replan.
        
        Different from `clear` method, the method would only clears memory and plans. 
        Besides, previous plan would be append to failed plans.
        """
        if len(self) > 0:
            text = f"Due to {inputs}, the replan is triggered. There are {len(self)} candidate games stored before, now they are cleared. Please replan the tool using."
        else:
            text = f"Due to {inputs}, the replan is triggered. There is no candidate games stored before. Please replan the tool using."
        self.failed_plans.append((self.plans, len(self.memory)))
        self.plans = []
        self.memory = []
        print(f"{self.__class__.__name__} is cleared for replan.")
        return text



    def __len__(self):
        return len(self.memory)
    

    def __str__(self) -> str:
        plans = f", ".join([f"{i}.{tool[0]}" for i, tool in enumerate(self.plans)])
        return f"Current tool using plan are: {plans}. Now there are {len(self)} candidate games in the buffer."
    

    def state(self) -> str:
        failed_plans = [f"Plan {n}: " + f", ".join([f"{i+1}.{tool}" for i, tool in enumerate(plan[0])]) 
                        + f". {plan[1]} candidate games are selected out under the plan." 
                        for n, plan in enumerate(self.failed_plans)]
        failed_plans = "\n".join(failed_plans)
        current_plan = f"; ".join([f"{i+1}. use {tool[0]}, obtain {tool[1]} candidate games" for i, tool in enumerate(self.plans)])
        state = f"\n```\n" \
                f"Buffer State\nFailed Plans: {failed_plans}\n" \
                f"Current Plan: {current_plan}\n" \
                f"Candidate Game Number: {len(self)}\n" \
                f"```\n"
        return state
