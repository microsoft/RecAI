import json
import random
from typing import *

import numpy as np
import pandas as pd

from pandasql import sqldf
from pandas.api.types import is_integer_dtype, is_bool_dtype, is_float_dtype, is_datetime64_dtype, is_object_dtype, is_categorical_dtype

from llm4crs.utils import raise_error, SentBERTEngine

_REQUIRED_COLUMNS = ['id', 'title']


def _pd_type_to_sql_type(col: pd.Series) -> str:
    res = ''
    if is_integer_dtype(col):
        res = 'integer'
    elif is_float_dtype(col):
        res = 'float'
    elif is_bool_dtype(col):
        res = 'boolean'
    elif is_datetime64_dtype(col):
        res = 'datetime'
    elif is_object_dtype(col) or is_categorical_dtype(col):
        res = 'string'
    else:
        res = 'string'
    return res


class BaseGallery:

    def __init__(self, fpath: str, column_meaning_file: str, name: str='Item_Information', columns: List[str]=None, sep: str=',', parquet_engine: str='pyarrow', 
                 fuzzy_cols: List[str]=['title'], categorical_cols: List[str]=['tags']) -> None:
        self.fpath = fpath
        self.name = name    # name of the table
        self.corups = self._read_file(fpath, columns, sep, parquet_engine)
        self._required_columns_validate()
        self.column_meaning = self._load_col_desc_file(column_meaning_file)

        self.categorical_col_values = {}
        for col in categorical_cols:
            _explode_df = self.corups.explode(col)[col]
            self.categorical_col_values[col] = _explode_df[~ _explode_df.isna()].unique()
            if isinstance(self.corups[col][0], str):
                pass
            else:
                self.corups[col] = self.corups[col].apply(lambda x: ', '.join(x))

        self.fuzzy_engine: Dict[str:SentBERTEngine] = {
            col : SentBERTEngine(self.corups[col].to_numpy(), self.corups['id'].to_numpy(), case_sensitive=False) if col not in categorical_cols
            else SentBERTEngine(self.categorical_col_values[col], np.arange(len(self.categorical_col_values[col])), case_sensitive=False) 
            for col in fuzzy_cols
        }
        # title as index
        self.corups_title = self.corups.set_index('title', drop=True)
        # id as index
        self.corups.set_index('id', drop=True, inplace=True)


    def __call__(self, sql: str, corups: pd.DataFrame=None, return_id_only: bool=True) -> List:
        """Search in corups with SQL query
        
        Args:
            sql: A sql query command.

        Returns:
            list: the result represents by id
        """
        try:
            if corups is None:
                res = sqldf(sql, {self.name: self.corups})    # all games
            else:
                res = sqldf(sql, {self.name: corups})   # games in buffer

            if return_id_only:
                res = res[self.corups.index.name].to_list()
            else:
                pass
            return res
        except Exception as e:
            print(e)
            return []


    def __len__(self) -> int:
        return len(self.corups)
        

    def info(self, remove_game_titles: bool=False):
        prefix = 'Table information:'
        table_name = f"Table Name: {self.name}"
        cols_info = "Column Names, Data Types and Column meaning:"
        cols_info += f"\n    - {self.corups.index.name}({_pd_type_to_sql_type(self.corups.index)}): {self.column_meaning[self.corups.index.name]}"
        for col in self.corups.columns:
            if remove_game_titles and 'title' in col:
                continue
            dtype = _pd_type_to_sql_type(self.corups[col])
            cols_info += f"\n    - {col}({dtype}): {self.column_meaning[col]}"
            if col == 'tags':
                _prefix = f" Such as [{', '.join(random.sample(self.categorical_col_values[col].tolist(), k=10))}]."
                cols_info += _prefix
            
            if dtype in {'float', 'datetime', 'integer'}:
                _min = self.corups[col].min()
                _max = self.corups[col].max()
                _mean = self.corups[col].mean()
                _median = self.corups[col].median()
                _prefix = f" Value ranges from {_min} to {_max}. The average value is {_mean}. The median is {_median}."
                cols_info += _prefix

        primary_key = f"Primary Key: {self.corups.index.name}"
        foreign_key = f"Foreign Key: None"
        res = ''
        for i, s in enumerate([table_name, cols_info, primary_key, foreign_key]):
            res += f"\n{i}. {s}"
        res = prefix + res
        return res


    def convert_id_2_info(self, id: Union[int, List[int], np.ndarray], col_names: Union[str, List[str]]=None) -> Union[Dict, List[Dict]]:
        """Given game id, get game informations.
        
        Args:
            - id: game ids. 
            - col_names: column names to be returned

        Returns:
            - information of given game ids, each game is formatted as a dict, whose key is column name.
        
        """
        if col_names is None:
            col_names = self.corups.columns
        else:
            if isinstance(col_names, str):
                col_names = [col_names]
            elif isinstance(col_names, list):
                pass
            else:
                raise_error(TypeError, "Not supported type for `col_names`.")

        if isinstance(id, int):
            items = self.corups.loc[id][col_names].to_dict()
        elif isinstance(id, list) or isinstance(id, np.ndarray):
            items = self.corups.loc[id][col_names].to_dict(orient='list')
        else:
            raise_error(TypeError, "Not supported type for `id`.")
    
        return items


    def convert_title_2_info(self, titles: Union[int, List[int], np.ndarray], col_names: Union[str, List[str]]=None) -> Union[Dict, List[Dict]]:
        """Given game title, get game informations.
        
        Args:
            - titles: game titles. Note that the game title must exist in the table. 
            - col_names: column names to be returned

        Returns:
            - information of given game titles, each game is formatted as a dict, whose key is column name.
        
        """
        if col_names is None:
            col_names = self.corups_title.columns
        else:
            if isinstance(col_names, str):
                col_names = [col_names]
            elif isinstance(col_names, list):
                pass
            else:
                raise_error(TypeError, "Not supported type for `col_names`.")

        if isinstance(titles, str) or (isinstance(titles, np.ndarray) and len(titles.shape)==0):
            items = self.corups_title.loc[titles][col_names].to_dict()
        elif isinstance(titles, list) or (isinstance(titles, np.ndarray) and len(titles.shape)>0):
            items = self.corups_title.loc[titles][col_names].to_dict(orient='list')
        else:
            raise_error(TypeError, "Not supported type for `titles`.")
    
        return items


    def _read_file(self, fpath: str, columns: List[str]=None, sep: str=',', parquet_engine: str='pyarrow') -> pd.DataFrame:
        if fpath.endswith('.csv') or fpath.endswith('.tsv'):
            df = pd.read_csv(fpath, sep=sep, names=columns)
        elif fpath.endswith('.ftr'):
            df = pd.read_feather(fpath)
        elif fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath, engine=parquet_engine)
        else:
            raise_error(TypeError, "Not support for such file type now.")
        if columns is not None:
            df = df[columns]
        print(f"Columns in item corups: {df.columns}.")
        return df


    def _load_col_desc_file(self, fpath: str) -> Dict:
        assert fpath.endswith('.json'), "Only support json file now."
        with open(fpath, 'r') as f:
            return json.load(f)

    
    def _required_columns_validate(self) -> None:
        for col in _REQUIRED_COLUMNS:
            if col not in self.corups.columns:
                raise_error(ValueError, f"`id` and `name` are required in item corups table but {col} not found, please check the table file `{self.fpath}`.")
            

    def fuzzy_match(self, value: Union[str, List[str]], col: str) -> Union[str, List[str]]:
        if col not in self.fuzzy_engine:
            raise_error(ValueError, f"Not support fuzzy search for column {col}")
        else:
            res = self.fuzzy_engine[col](value, return_doc=True, topk=1) 
            res = np.squeeze(res, axis=-1)
            return res


    
if __name__ == '__main__':
    from llm4crs.environ_variables import *
    gallery = BaseGallery(GAME_INFO_FILE, column_meaning_file=TABLE_COL_DESC_FILE)
    print(gallery.info())

    sql = f"SELECT * FROM {gallery.name} WHERE id<5"

    print("SQL result: ", gallery(sql))

    res = gallery.fuzzy_match(['Call of Duty', 'Call Duty2'], 'title')
    print(res)
    res = gallery.fuzzy_match('Call of Duty', 'title')
    print(res)

    print('Test end.')