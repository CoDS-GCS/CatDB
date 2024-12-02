import warnings
from typing import Any, Dict, List, Optional

from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader

from .configs import Config
from .formatter import format_report
from .report import Report

__all__ = ["create_report"]

ENV_LOADER = Environment(
    loader=PackageLoader("catdb", "ui/templates"),
)


def create_report(
    catalog,
    pipegen,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    title: Optional[str] = "CatDB Report",
    mode: Optional[str] = "basic",
    progress: bool = True,
) -> Report:

    _suppress_warnings()
    cfg = Config.from_dict(display, config)
    context = {
        "resources": INLINE.render(),
        "title": title,
        "components": format_report(catalog[0], pipegen, cfg, mode, progress),
    }
    template_base = ENV_LOADER.get_template("base.html")
    report = template_base.render(context=context)
    return Report(report)


def _suppress_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        "The default value of regex will change from True to False in a future version",
        category=FutureWarning,
    )
