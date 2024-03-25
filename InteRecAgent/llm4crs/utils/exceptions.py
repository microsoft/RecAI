# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

_ERR_PREFIX = 'LLM4CRS Error'

def raise_error(err_cls, msg: str):
    raise err_cls(f"{_ERR_PREFIX}: {msg}")


__all__ = [
    "raise_error"
]