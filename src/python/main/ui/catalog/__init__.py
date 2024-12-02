from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..configs import Config
from ..container import Container
from .render import render_catalog


# __all__ = ["create_catalog_report"]
__all__ = ["compute_catalog"]

# def create_catalog_report(catalog: list,
#                           config: Optional[Dict[str, Any]] = None,
#                           display: Optional[List[str]] = None) -> Container:
#
#     cfg = Config.from_dict(display, config)
#     to_render = render_catalog(catalog=catalog[0], cfg=cfg)
#     return Container(to_render, 'grid', cfg)


def compute_catalog(catalog: list, cfg: Config,):
    return  render_catalog(catalog, cfg)