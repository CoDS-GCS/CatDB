from .catalog import compute_catalog
from .pipegen import compute_pipegen

from typing import Any, Dict, Optional
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

def _format_pipegen(pipegen, cfg: Config) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    res['pipegen'] = compute_pipegen(pipegen, cfg)
    return res

def format_basic(catalog, pipegen, cfg: Config) -> Dict[str, Any]:
    setattr(getattr(cfg, "plot"), "report", True)
    res_catalog = _format_catalog(catalog, cfg)
    res_pipegen = _format_pipegen(pipegen, cfg)

    res = {**res_catalog, **res_pipegen}
    return res