from .catalog import compute_catalog

from typing import Any, Dict, List, Optional, Tuple, Union
from .configs import Config


def format_report(
    catalog,
    pipegen,
    cfg: Config,
    mode: Optional[str],
    progress: bool = True,
) -> Dict[str, Any]:

    comps = format_basic(catalog, pipegen, cfg)
    return comps


def _format_catalog(catalog, cfg: Config) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    res['catalog'] = compute_catalog(catalog, cfg)
    return res


def format_basic(catalog, pipegen, cfg: Config) -> Dict[str, Any]:
    setattr(getattr(cfg, "plot"), "report", True)
    res_variables = _format_catalog(catalog, cfg)
    res = {**res_variables}
    return res