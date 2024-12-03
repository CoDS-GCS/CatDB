from ..configs import Config
from .render import render_pipegen


__all__ = ["compute_pipegen"]


def compute_pipegen(pipegen: str, cfg: Config,):
    return render_pipegen(pipegen, cfg)