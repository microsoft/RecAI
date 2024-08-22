"""
The following code is modified from
https://github.com/microsoft/UniRec/blob/main/unirec/model/sequential/sasrec.py
"""

from models.base_rec import BaseRecommender

class MF(BaseRecommender):    
    def __init__(self, config):
        super(MF, self).__init__(config)
        ## every thing should be ready from the BaseRecommender class


