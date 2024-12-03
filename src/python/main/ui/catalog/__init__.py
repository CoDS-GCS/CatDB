from ..configs import Config
from .render import render_catalog


__all__ = ["compute_catalog"]


def compute_catalog(catalog: list, cfg: Config,):
    return  render_catalog(catalog, cfg)