from .compute import compute_results
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from bokeh.embed import components
from ..configs import Bar
from ..utils import _format_axis
from bokeh.models import (
    ColumnDataSource,
    CustomJSHover,
    HoverTool,
)
from bokeh.plotting import figure

from ..configs import Config, Box
from ..palette import CATEGORY20

from ..utils import relocate_legend

__all__ = ["render_pipegen"]


def render_pipegen(pipegen, cfg: Config) -> Dict[str, Any]:
    plot_width = cfg.plot.width if cfg.plot.width is not None else 300
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300

    comp_res = compute_results(pipegen)
    df_runtime = comp_res["df_runtime"]
    df_cost = comp_res["df_cost"]
    df_error = comp_res["df_error"]
    max_runtime = max(df_runtime["time_total"])
    max_token = max(df_cost["Error Tokens"] + df_cost["Prompt Tokens"])
    df_runtime = df_runtime[['Pipeline Generation', 'Pipeline Verify', 'Pipeline Execution']]

    performance_1, performance_2 = None, None
    if pipegen["task_type"] == "multiclass":
        df_auc_ovr = comp_res["performance_auc_ovr"]
        df_log_loss = comp_res["performance_log_loss"]
        performance_1 = box_viz(df=df_auc_ovr, plot_width=plot_width, plot_height=plot_height, box=cfg.box,
                                          ylabel="AUC-ovr [%]", x="Train", title="Area Under the Curve (AUC)")

        performance_2  = box_viz(df=df_log_loss, plot_width=plot_width, plot_height=plot_height, box=cfg.box,
                                          ylabel="Log Loss Value", x="Train", title="Log Loss")
    elif pipegen["task_type"] == "binary":
        df_auc = comp_res["performance_auc"]
        df_f1_score = comp_res["performance_f1_score"]
        performance_1 = box_viz(df=df_auc, plot_width=plot_width, plot_height=plot_height, box=cfg.box,
                                          ylabel="AUC [%]", x="Train", title="Area Under the Curve (AUC)")

        performance_2  = box_viz(df=df_f1_score, plot_width=plot_width, plot_height=plot_height, box=cfg.box,
                                          ylabel="Score [%]", x="Train", title="F1 Score")

    elif pipegen["task_type"] == "regression":
        df_r_squared= comp_res["performance_r_squared"]
        df_rmse = comp_res["performance_rmse"]
        performance_1 = box_viz(df=df_r_squared, plot_width=plot_width, plot_height=plot_height, box=cfg.box,
                                ylabel="R-squared [%]", x="Train", title="R-Squared")

        performance_2 = box_viz(df=df_rmse, plot_width=plot_width, plot_height=plot_height, box=cfg.box,
                                ylabel="RMSE [%]", x="Train", title="Root Mean Square Error (RMSE)")



    fig_runtime = render_bar_runtime_chart(max_runtime, df_runtime, "linear", 500, plot_height, "Time [second]", "#Iteration")
    fig_cost = render_bar_cost_chart(max_token, df_cost, "linear", 500, plot_height, "Token Count", "#Iteration")
    fig_error = bar_viz(df_error, len(df_error), "count", plot_width * 2, plot_height, True,
                               cfg.bar, "Error Type", "")

    res: Dict[str, Any] = {
        "runtime": {"fig": components(fig_runtime), "title": "Pipeline Runtime"},
        "performance_1": {"fig": components(performance_1), "title": ""},
        "performance_2": {"fig": components(performance_2), "title": ""},
        "cost": {"fig": components(fig_cost), "title": "Pipeline Cost"},
        "error": {"fig": components(fig_error), "title": "Pipeline Error"},
    }

    return res


def render_bar_chart(nrows: int, df, yscale: str, plot_width: int, plot_height: int, ylabel: str, title: str) -> figure:
    if len(df) > 20:
        plot_width = 28 * len(df)

    fig = figure(
        x_range=list(df.index),
        y_range=[0, nrows],
        width=plot_width,
        height=plot_height,
        y_axis_type=yscale,
        toolbar_location=None,
        tools=[],
        title=title,
    )

    rend = fig.vbar_stack(
        stackers=df.columns,
        x="index",
        width=0.9,
        color=[CATEGORY20[0], CATEGORY20[2]],
        source=df,
        legend_label=list(df.columns),
    )
    # hover tool with count and percent
    formatter = CustomJSHover(
        args=dict(source=ColumnDataSource(df)),
        code="""
        const columns = Object.keys(source.data)
        const cur_bar = special_vars.data_x - 0.5
        var ttl_bar = 0
        for (let i = 0; i < columns.length; i++) {
            if (columns[i] != 'index'){
                ttl_bar = ttl_bar + source.data[columns[i]][cur_bar]
            }
        }
        const cur_val = source.data[special_vars.name][cur_bar]
        return (cur_val/ttl_bar * 100).toFixed(2)+'%';
    """,
    )
    for i, val in enumerate(df.columns):
        hover = HoverTool(
            tooltips=[
                ("Column", "@index"),
                (f"{val} count", "@$name"),
                (f"{val} percent", "@{%s}{custom}" % rend[i].name),
            ],
            formatters={"@{%s}" % rend[i].name: formatter},
            renderers=[rend[i]],
        )
        fig.add_tools(hover)

    fig.yaxis.axis_label = ylabel
    tweak_figure(fig)
    relocate_legend(fig, "left")

    return fig


def render_bar_runtime_chart(max_time: int, df, yscale: str, plot_width: int, plot_height: int, ylabel: str, xlabel: str) -> figure:

    colors = [CATEGORY20[0], CATEGORY20[2], CATEGORY20[4]]
    fig = figure(
        x_range=[str(i) for i in list(df.index)],
        y_range=[0, max_time],
        width=plot_width,
        height=plot_height,
        y_axis_type=yscale,
        toolbar_location=None,
        tools=[],
        title="",
    )

    rend = fig.vbar_stack(
        stackers=df.columns,
        x="index",
        width=0.9,
        color=colors,
        source=df,
        legend_label=list(df.columns),
    )
    # hover tool with count and percent
    formatter = CustomJSHover(
        args=dict(source=ColumnDataSource(df)),
        code="""
        const columns = Object.keys(source.data)
        const cur_bar = special_vars.data_x - 0.5
        var ttl_bar = 0
        for (let i = 0; i < columns.length; i++) {
            if (columns[i] != 'index'){
                ttl_bar = ttl_bar + source.data[columns[i]][cur_bar]
            }
        }
        const cur_val = source.data[special_vars.name][cur_bar]
        return (cur_val/ttl_bar * 100).toFixed(2)+'%';
    """,
    )
    for i, val in enumerate(df.columns):
        hover = HoverTool(
            tooltips=[
                ("#Iteration", "@index"),
                (f"{val} time", "@$name"),
            ],
            formatters={"@{%s}" % rend[i].name: formatter},
            renderers=[rend[i]],
        )
        fig.add_tools(hover)

    fig.yaxis.axis_label = ylabel
    fig.xaxis.axis_label = xlabel
    tweak_figure(fig)
    relocate_legend(fig, "left")

    return fig


def render_bar_cost_chart(max_cost: int, df, yscale: str, plot_width: int, plot_height: int, ylabel: str, xlabel: str) -> figure:
    colors = [CATEGORY20[0], CATEGORY20[2]]
    fig = figure(
        x_range=[str(i) for i in list(df.index)],
        y_range=[0, max_cost],
        width=plot_width,
        height=plot_height,
        y_axis_type=yscale,
        toolbar_location=None,
        tools=[],
        title="",
    )

    rend = fig.vbar_stack(
        stackers=df.columns,
        x="index",
        width=0.5,
        color=colors,
        source=df,
        legend_label=list(df.columns),
    )
    # hover tool with count and percent
    formatter = CustomJSHover(
        args=dict(source=ColumnDataSource(df)),
        code="""
        const columns = Object.keys(source.data)
        const cur_bar = special_vars.data_x - 0.5
        var ttl_bar = 0
        for (let i = 0; i < columns.length; i++) {
            if (columns[i] != 'index'){
                ttl_bar = ttl_bar + source.data[columns[i]][cur_bar]
            }
        }
        const cur_val = source.data[special_vars.name][cur_bar]
        return (cur_val/ttl_bar * 100).toFixed(2)+'%';
    """,
    )
    for i, val in enumerate(df.columns):
        hover = HoverTool(
            tooltips=[
                ("#Iteration", "@index"),
                (f"{val}", "@$name"),
            ],
            formatters={"@{%s}" % rend[i].name: formatter},
            renderers=[rend[i]],
        )
        fig.add_tools(hover)

    fig.yaxis.axis_label = ylabel
    fig.xaxis.axis_label = xlabel
    tweak_figure(fig)
    relocate_legend(fig, "left")

    return fig


def box_viz(
    df: pd.DataFrame,
    x: str,
    plot_width: int,
    plot_height: int,
    box: Box,
    y: Optional[str] = None,
    ttl_grps: Optional[int] = None,
    ylabel: str = "",
    title: str = ''
) -> figure:

    width = 0.5
    df["x0"], df["x1"] = df.index + 0.2, df.index + 0.8
    fig = figure(
        width=plot_width,
        height=plot_height,
        title=title,
        toolbar_location=None,
        x_range=df["grp"],
    )
    low = fig.segment(x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df)
    ltail = fig.segment(x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df)
    lbox = fig.vbar(
        x="grp",
        width=width,
        top="q2",
        bottom="q1",
        fill_color=box.color,
        line_color="black",
        source=df,
    )
    ubox = fig.vbar(
        x="grp",
        width=width,
        top="q3",
        bottom="q2",
        fill_color=box.color,
        line_color="black",
        source=df,
    )
    utail = fig.segment(x0="grp", x1="grp", y0="uw", y1="q3", line_color="black", source=df)
    upw = fig.segment(x0="x0", y0="uw", x1="x1", y1="uw", line_color="black", source=df)

    df.loc[df["otlrs"].isna(), "otlrs"] = pd.Series(
        [[]] * df["otlrs"].isna().sum(), dtype=np.float64
    ).values
    otlrs = [otl for otls in df["otlrs"] for otl in otls]
    if otlrs:
        gps = [grp for grp, ols in zip(df["grp"], df["otlrs"]) for _ in range(len(ols))]
        circ = fig.circle(
            x=gps,
            y=otlrs,
            radius=0.01,
            line_color="black",
            color="black",
            fill_alpha=0.6,
        )
        fig.add_tools(
            HoverTool(
                renderers=[circ],
                tooltips=[("Outlier", "@y")],
            )
        )
    tooltips = [
        ("Upper Whisker", "@uw"),
        ("Upper Quartile", "@q3"),
        ("Median", "@q2"),
        ("Lower Quartile", "@q1"),
        ("Lower Whisker", "@lw"),
    ]
    if y:
        lbl = f"{x}" if ttl_grps else "Bin"
        tooltips.insert(0, (lbl, "@grp"))
    fig.add_tools(
        HoverTool(
            renderers=[upw, utail, ubox, lbox, ltail, low],
            tooltips=tooltips,
        )
    )
    tweak_figure_bar(fig, "box")
    fig.xaxis.axis_label = ""
    fig.yaxis.axis_label = ylabel
    return fig


def tweak_figure(fig: figure) -> figure:
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = "9pt"
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = np.pi / 3

    return fig


def tweak_figure_bar(fig: figure, ptype: Optional[str] = None, show_yticks: bool = False,
                     max_lbl_len: int = 15, ) -> None:
    fig.axis.major_label_text_font_size = "9pt"
    fig.title.text_font_size = "10pt"
    fig.axis.minor_tick_line_color = "white"
    if ptype in ["pie", "qq", "heatmap"]:
        fig.ygrid.grid_line_color = None
    if ptype in ["bar", "pie", "hist", "kde", "qq", "heatmap", "line"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["bar", "hist", "line"] and not show_yticks:
        fig.ygrid.grid_line_color = None
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
    if ptype in ["bar", "nested", "stacked", "heatmap", "box"]:
        fig.xaxis.major_label_orientation = np.pi / 3
        # fig.xaxis.formatter = FuncTickFormatter(
        #     code="""
        #     if (tick.length > %d) return tick.substring(0, %d-2) + '...';
        #     else return tick;
        # """
        #          % (max_lbl_len, max_lbl_len)
        # )
    if ptype in ["nested", "stacked", "box"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["nested", "stacked"]:
        fig.y_range.start = 0
        fig.x_range.range_padding = 0.03
    if ptype in ["line", "boxnum"]:
        fig.min_border_right = 20
        fig.xaxis.major_label_standoff = 7
        fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None


def _format_ticks(ticks: List[float]) -> List[str]:
    """
    Format the tick values
    """
    formatted_ticks = []
    for tick in ticks:  # format the tick values
        before, after = f"{tick:e}".split("e")
        if float(after) > 1e15 or abs(tick) < 1e4:
            formatted_ticks.append(str(tick))
            continue
        mod_exp = int(after) % 3
        factor = 1 if mod_exp == 0 else 10 if mod_exp == 1 else 100
        value = np.round(float(before) * factor, len(str(before)))
        value = int(value) if value.is_integer() else value
        if abs(tick) >= 1e12:
            formatted_ticks.append(str(value) + "T")
        elif abs(tick) >= 1e9:
            formatted_ticks.append(str(value) + "B")
        elif abs(tick) >= 1e6:
            formatted_ticks.append(str(value) + "M")
        elif abs(tick) >= 1e4:
            formatted_ticks.append(str(value) + "K")

    return formatted_ticks


def bar_viz(df: pd.DataFrame, ttl_grps: int, col: str, plot_width: int, plot_height: int, show_yticks: bool,
            bar_cfg: Bar,
            tooltip_title: str, title: str) -> figure:
    tooltips = [(tooltip_title, "@index"), ("Count", f"@{{{col}}}"), ("Percent", "@pct{0.2f}%")]
    if show_yticks:
        if len(df) > 10:
            plot_width = 28 * len(df)
    fig = figure(
        width=plot_width,
        height=plot_height,
        title=title,
        toolbar_location=None,
        tooltips=tooltips,
        tools="hover",
        x_range=list(df.index),
        y_axis_type=bar_cfg.yscale,
    )
    fig.vbar(
        x="index",
        width=0.9,
        top=col,
        fill_color=bar_cfg.color,
        line_color=bar_cfg.color,
        bottom=0.01,
        source=df,
    )
    tweak_figure_bar(fig, "bar", show_yticks)
    fig.yaxis.axis_label = "Count"
    if ttl_grps > len(df):
        fig.xaxis.axis_label = f"Top {len(df)} of {ttl_grps} {col}"
        fig.xaxis.axis_label_standoff = 0
    if show_yticks and bar_cfg.yscale == "linear":
        _format_axis(fig, 0, df[col].max(), "y")
    return fig


def _format_values(key: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        # if value is a time
        return str(value)

    if "Memory" in key:
        # for memory usage
        ind = 0
        unit = dict(enumerate(["B", "KB", "MB", "GB", "TB"], 0))
        while value > 1024:
            value /= 1024
            ind += 1
        return f"{value:.1f} {unit[ind]}"

    if (value * 10) % 10 == 0:
        # if value is int but in a float form with 0 at last digit
        value = int(value)
        if abs(value) >= 1000000:
            return f"{value:.5g}"
    elif abs(value) >= 1000000 or abs(value) < 0.001:
        value = f"{value:.5g}"
    elif abs(value) >= 1:
        # eliminate trailing zeros
        pre_value = float(f"{value:.4f}")
        value = int(pre_value) if (pre_value * 10) % 10 == 0 else pre_value
    elif 0.001 <= abs(value) < 1:
        value = f"{value:.4g}"
    else:
        value = str(value)

    if "%" in key:
        # for percentage, only use digits before notation sign for extreme small number
        value = f"{float(value):.1%}"
    return str(value)


def format_cat_stats(
        stats: Dict[str, Any],
        len_stats: Dict[str, Any],
        letter_stats: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    """
    Format categorical statistics
    """
    ov_stats = {
        "Approximate Distinct Count": stats["nuniq"],
        "Approximate Unique (%)": stats["nuniq"] / stats["npres"],
        "Missing": stats["nrows"] - stats["npres"],
        "Missing (%)": 1 - stats["npres"] / stats["nrows"],
        "Memory Size": stats["mem_use"],
    }
    sampled_rows = ("1st row", "2nd row", "3rd row", "4th row", "5th row")
    smpl = dict(zip(sampled_rows, stats["first_rows"]))

    return {
        "Overview": {k: _format_values(k, v) for k, v in ov_stats.items()},
        "Length": {k: _format_values(k, v) for k, v in len_stats.items()},
        "Sample": {k: f"{v[:18]}..." if len(v) > 18 else v for k, v in smpl.items()},
        "Letter": {k: _format_values(k, v) for k, v in letter_stats.items()},
    }
