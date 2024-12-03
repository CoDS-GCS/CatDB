from typing import Any, Dict, Optional
from .compute import compute_statistics

import numpy as np
import pandas as pd
from bokeh.embed import components
from ..configs import  Bar
from ..utils import _format_axis
from bokeh.models import (
    ColumnDataSource,
    CustomJSHover,
    HoverTool)
from bokeh.plotting import figure

from ..configs import Config
from ..palette import CATEGORY20, BLUES256, REDS256, PURLPLES256, ORANGES256, TURBO256, CIVIDIS256
from ..utils import relocate_legend

__all__ = ["render_catalog"]


def render_catalog(catalog, cfg: Config) -> Dict[str, Any]:
    df_overview = compute_statistics(catalog=catalog)
    schema_info = catalog.schema_info
    profile_info = catalog.profile_info

    plot_width = cfg.plot.width if cfg.plot.width is not None else 200
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300

    nrows = catalog.nrows
    cols = schema_info.keys()
    present = []
    missing = []
    distinct = []
    all_cat_cols = dict()
    max_col_count = 0
    categorical_col_count = 0
    numerical_col_count = 0
    str_col_count = 0

    for col in cols:
        present.append(profile_info[col].total_values_count)
        missing.append(profile_info[col].missing_values_count)
        distinct.append(profile_info[col].distinct_values_count)
        if profile_info[col].is_categorical:
            all_cat_cols[col] = profile_info[col].categorical_values_ratio
            max_col_count = max(max_col_count, len(profile_info[col].categorical_values))
            categorical_col_count += 1
        elif profile_info[col].short_data_type == 'str':
            str_col_count += 1
        else:
            numerical_col_count += 1

    df_missing = pd.DataFrame({"Present": present, "Missing": missing}, index=cols)
    df_distinct = pd.DataFrame({"Distinct": distinct, 'pct': [d / nrows for d in distinct]}, index=cols)
    df_cat = pd.DataFrame(columns=[f'Categorical Item {i}' for i in range(1, max_col_count + 1)],
                          index=list(all_cat_cols.keys()))
    df_feature_type = pd.DataFrame({"count": [categorical_col_count, numerical_col_count, str_col_count],
                                    'pct': [categorical_col_count / len(cols), numerical_col_count / len(cols),
                                            str_col_count / len(cols)]},
                                   index=["Categorical", "Numerical", "Sentence"])

    for col in all_cat_cols.keys():
        values = list(all_cat_cols[col].values())
        values = sorted(values, key=int, reverse=True)
        cols = [f'Categorical Item {i}' for i in range(1, len(all_cat_cols[col].keys()) + 1)]
        df_cat.loc[col, cols] = values

    df_cat = df_cat.astype(float).fillna(0)
    fig_overview = bar_viz(df_overview, len(df_overview), "count", plot_width, plot_height, True, cfg.bar, "Data Type", "")
    fig_feature_type = bar_viz(df_feature_type, len(df_feature_type), "count", plot_width, plot_height, True, cfg.bar, "Feature Type", "")

    fig_missing = render_bar_chart(nrows, df_missing, "linear", plot_width, plot_height, "Samples",  "")
    fig_missing.frame_width = plot_width

    fig_distinct = bar_viz(df_distinct, len(df_distinct), "Distinct", plot_width, plot_height, True, cfg.bar, "Column", "")
    fig_distinct.frame_width = plot_width

    fig_cat = render_bar_categorical_chart(nrows, df_cat, "linear", int(plot_width * 1.5), plot_height, "Frequency","")

    res: Dict[str, Any] = {
        "overview": {"fig": components(fig_overview), "title": "Dataset Overview"},
        "feature_type": {"fig": components(fig_feature_type), "title": "Feature Types"},
        "missing": {"fig": components(fig_missing), "title": "Missing Value"},
        "distinct": {"fig": components(fig_distinct), "title": "Distinct Values"},
        "cat": {"fig": components(fig_cat), "title": "Categorical Column and Values"}
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
        width=0.5,
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


def render_bar_categorical_chart(nrows: int, df, yscale: str, plot_width: int, plot_height: int, ylabel: str,
                                 title: str) -> figure:
    if len(df) > 20:
        plot_width = 28 * len(df)

    base_colors = []
    for i in range(60, 256, 10):
        base_colors.append(BLUES256[i])
        base_colors.append(ORANGES256[i])
        base_colors.append(REDS256[i])
        base_colors.append(PURLPLES256[i])
        base_colors.append(TURBO256[i])
        base_colors.append(CIVIDIS256[i])

    for i in range(60, 256, 7):
        base_colors.append(BLUES256[i])
        base_colors.append(ORANGES256[i])
        base_colors.append(REDS256[i])
        base_colors.append(PURLPLES256[i])
        base_colors.append(TURBO256[i])
        base_colors.append(CIVIDIS256[i])

    colors = [base_colors[c] for c in range(0, len(df.columns))]
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
    fig.legend.visible = False

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
        width=0.5,
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
