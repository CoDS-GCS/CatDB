from .catalog import compute_catalog
from .pipegen import compute_pipegen

from typing import Any, Dict
from .configs import Config


def format_report(
        catalog,
        pipegen,
        cfg: Config
) -> Dict[str, Any]:
    comps = format_basic(catalog, pipegen, cfg)
    return comps


def _format_catalog(catalog, cfg: Config) -> Dict[str, Any]:
    res = compute_catalog(catalog, cfg)
    return res


def _format_pipegen(pipegen, cfg: Config) -> Dict[str, Any]:
    res = compute_pipegen(pipegen, cfg)
    return res


def format_basic(catalog, pipegen, cfg: Config) -> Dict[str, Any]:
    setattr(getattr(cfg, "plot"), "report", True)
    res: Dict[str, Any] = {}
    if catalog is not None:
        res["catalog"] = _format_catalog(catalog, cfg)
    if pipegen is not None:
        res["pipegen"] = _format_pipegen(pipegen, cfg)
    return res
