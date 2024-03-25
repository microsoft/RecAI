# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os


_NAME = "Game Candidates Cleanup Tool"
_DESC = "The tool is useful when you need to cleanup candidate games and \
replan tool using order due to there are too few candidate games to recommend. \
The tool would cleanup all candidate games selected in previous tool using. The input of the tool should be None."


class CandidateCleanupTool:

    def __init__(self) -> None:
        self.name = _NAME
        self.desc = _DESC


    def run(self, inputs: str) -> str:
        candidates = eval(os.environ.get("llm4crs_candidates", "[]"))
        if len(candidates) > 0:
            text = f"There are {len(candidates)} candidate games stored, now they are cleared. Please replan the tool using."
        else:
            text = f"There is no candidate games stored. Please replan the tool using."
        return text
