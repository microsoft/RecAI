# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import random
from loguru import logger

from llm4crs.corups import BaseGallery
from llm4crs.buffer import CandidateBuffer
from llm4crs.utils.sql import extract_columns_from_where



class SQLSearchTool:

    def __init__(self, name: str, desc: str, item_corups: BaseGallery, buffer: CandidateBuffer, max_candidates_num: int=None) -> None:
        self.item_corups = item_corups
        self.buffer = buffer
        self.name = name
        self.desc = desc
        self.max_candidates_num = max_candidates_num


    def run(self, inputs: str) -> str:
        # candidates = eval(os.environ.get("llm4crs_candidates", "[]"))
        info = ""
        candidates = self.buffer.get()
        if len(candidates) > 0:
            info += f"Before {self.name}: There are {len(candidates)} candidates in buffer. \n"
            corups = self.item_corups.corups.loc[candidates]
        else:
            info += f"Before {self.name}: There are {len(candidates)} candidates in buffer. Stop execution. \n"
            return info
        
        logger.debug(f"\nSQL from AGI: {inputs}")
        try:
            inputs = self.rewrite_sql(inputs)
            logger.debug(f"Rewrite SQL: {inputs}")
            info += f"{self.name}: The input SQL is rewritten as {inputs} because some {list(self.item_corups.categorical_col_values.keys())} are not existing. \n"
        except Exception as e:
            logger.exception(e)
            info += f"{self.name}: something went wrong in execution, the tool is broken for current input. The candidates are not modified.\n"
            return info

        try:
            candidates = self.item_corups(inputs, corups=corups)    # list of ids
            n = len(candidates)
            _info = f"After {self.name}: There are {n} eligible items. "
            if self.max_candidates_num is not None:
                if len(candidates) > self.max_candidates_num:
                    if "order" in inputs.lower():
                        candidates = candidates[: self.max_candidates_num]
                        _info += f"Select the first {self.max_candidates_num} items from all eligible items ordered by the SQL. "
                    else:
                        candidates = random.sample(candidates, k=self.max_candidates_num)
                        _info += f"Random sample {self.max_candidates_num} items from all eligible items. "
                else:
                    pass
            else:
                pass

            info += _info
            self.buffer.push(self.name, candidates)
        except Exception as e:
            logger.debug(e)
            candidates = []
            info = f"{self.name}: something went wrong in execution, the tool is broken for current input. The candidates are not modified."

        self.buffer.track(self.name, inputs, info)

        # suffix = f"{len(candidates)} candidate games are selected with SQL command {inputs}. Those candidate games are stored and visible to other tools. Now you need to take the next action."
        # return  f"Here are candidates id searched with the SQL command: [{','.join(map(str, candidates))}]."
        logger.debug(f"{info}")
        return info


    def rewrite_sql(self, sql: str) -> str:
        """Rewrite SQL command using fuzzy search"""
        sql = re.sub(r'\bFROM\s+(\w+)\s+WHERE', f'FROM {self.item_corups.name} WHERE', sql, flags=re.IGNORECASE)
        
        # groudning cols
        cols = extract_columns_from_where(sql)
        existing_cols = set(self.item_corups.column_meaning.keys())
        col_replace_dict = {}
        for col in cols:
            if col not in existing_cols:
                mapped_col = self.item_corups.fuzzy_match(col, 'sql_cols')
                col_replace_dict[col] = f"{mapped_col}"
        for k, v in col_replace_dict.items():
            sql = sql.replace(k, v)

        # grounding categorical values
        pattern = r"([a-zA-Z0-9_]+) (?:NOT )?LIKE '\%([^\%]+)\%'" 
        res = re.findall(pattern, sql)
        replace_dict = {}
        for col, value in res:
            if col not in self.item_corups.fuzzy_engine:
                continue
            replace_value = str(self.item_corups.fuzzy_match(value, col))
            replace_value = replace_value.replace("'", "''")    # escaping string for sqlite
            replace_dict[f"%{value}%"] = f"%{replace_value}%"

        for k, v in replace_dict.items():
            sql = sql.replace(k, v)
        return sql
