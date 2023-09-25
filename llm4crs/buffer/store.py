# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from llm4crs.corups.base import BaseGallery

_NAME = "Game Candidates Storing Tool"
_DESC = """The tool is useful when human give candidate game names in conversation history. \
For example, "Please select the most suitable game from those games". \
You could use the tool to store candidate games given by human to buffer for other tools to further filter and rank. \
The input of the tool should be a list of game names split by comma, such as "game1, game2, game3". """

class CandidateStoreTool:
    """used to put candidate games extracted from conversation to data bus"""

    def __init__(self, item_corups: BaseGallery) -> None:
        self.name = _NAME
        self.desc = _DESC
        self.item_corups = item_corups


    def run(self, inputs: str) -> str:
        try:
            games = [x.strip() for x in inputs.split(',')]
            titles = self.item_corups.fuzzy_match(games, 'title')
            game_ids = self.item_corups.convert_title_2_info(titles, col_names='id')['id']
            os.environ['llm4crs_candidates'] = f"[{','.join([str(x) for x in game_ids])}]"
            return f"{len(game_ids)} candidate games are given by human in conversation and stored in buffer. " \
                "Those games are visible to other tools to be further filtered and ranked."
        except Exception as e:
            print(e)
            return "Storing games failed, please retry."