# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json

_domain = os.environ.get('DOMAIN', 'game')
__filepath = os.path.dirname(__file__)
__domain_resource_dir = os.path.abspath(os.path.join(__filepath, f'../resources/{_domain}'))
UNIREC_DIR = os.path.abspath(os.path.join(__filepath, '../UniRec/'))

with open(os.path.join(__domain_resource_dir, 'settings.json')) as f:
    environ = json.load(f)

GAME_INFO_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['GAME_INFO_FILE']))
TABLE_COL_DESC_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['TABLE_COL_DESC_FILE']))
MODEL_CKPT_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['MODEL_CKPT_FILE']))
ITEM_SIM_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['ITEM_SIM_FILE']))
USE_COLS = environ['USE_COLS']
CATEGORICAL_COLS = environ['CATEGORICAL_COLS']

__all__ = ['UNIREC_DIR', 'GAME_INFO_FILE', 'TABLE_COL_DESC_FILE', 'MODEL_CKPT_FILE', 'ITEM_SIM_FILE', 'USE_COLS', 'CATEGORICAL_COLS']