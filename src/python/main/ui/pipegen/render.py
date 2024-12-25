from .compute import compute_results
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from bokeh.embed import components
import json
import  re
from ..configs import Bar, tweak_figure, tweak_figure_bar, format_ticks, format_values, format_cat_stats
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
    plot_width_performance = int(plot_width * 0.75)
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300

    comp_res = compute_results(pipegen)
    df_runtime = comp_res["df_runtime"]
    df_cost = comp_res["df_cost"]
    df_error = comp_res["df_error"]
    max_runtime = max(df_runtime["time_total"])
    max_token = max(df_cost["Error Tokens"] + df_cost["Prompt Tokens"])
    df_runtime = df_runtime[['Generation', 'Verify', 'Execution']]

    performance_1, performance_2 = None, None
    if pipegen["task_type"] == "multiclass":
        df_auc_ovr = comp_res["performance_auc_ovr"]
        df_log_loss = comp_res["performance_log_loss"]
        performance_1 = box_viz(df=df_auc_ovr, plot_width=plot_width_performance, plot_height=plot_height, box=cfg.box,
                                ylabel="AUC-ovr [%]", x="Train", title="AUC OVR", ratio=100)

        performance_2 = box_viz(df=df_log_loss, plot_width=plot_width_performance, plot_height=plot_height, box=cfg.box,
                                ylabel="Log Loss Value", x="Train", title="Log Loss", ratio=100)
    elif pipegen["task_type"] == "binary":
        df_auc = comp_res["performance_auc"]
        df_f1_score = comp_res["performance_f1_score"]
        performance_1 = box_viz(df=df_auc, plot_width=plot_width_performance, plot_height=plot_height, box=cfg.box,
                                ylabel="AUC [%]", x="Train", title="AUC", ratio=100)

        performance_2 = box_viz(df=df_f1_score, plot_width=plot_width_performance, plot_height=plot_height, box=cfg.box,
                                ylabel="Score [%]", x="Train", title="F1 Score", ratio=100)

    elif pipegen["task_type"] == "regression":
        df_r_squared = comp_res["performance_r_squared"]
        df_rmse = comp_res["performance_rmse"]
        performance_1 = box_viz(df=df_r_squared, plot_width=plot_width_performance, plot_height=plot_height,
                                box=cfg.box, ylabel="R-squared [%]", x="Train", title="R-Squared", ratio=100)

        performance_2 = box_viz(df=df_rmse, plot_width=plot_width_performance, plot_height=plot_height, box=cfg.box,
                                ylabel="RMSE [%]", x="Train", title="RMSE")

    fig_runtime = render_bar_runtime_chart(max_runtime, df_runtime, "linear", plot_width, plot_height, "Time [second]",
                                           "#Iteration")
    fig_cost = render_bar_cost_chart(max_token, df_cost, "linear", plot_width, plot_height, "Token Count", "#Iteration")

    if len(df_error) > 0:
        error_count = len(df_error)
        item_width = 25
        bar_width = 0.5
        if 5 < error_count <= 10:
            bar_width = 0.3
        elif error_count > 10:
            bar_width = 0.2

        error_plot_width = 150 + item_width * error_count
        fig_error = bar_viz(df_error, len(df_error), "count", error_plot_width, bar_width, plot_height, True,
                            cfg.bar, "Error Type", "")
        error_result = components(fig_error)
        has_error = True
    else:
        error_result = None
        has_error = False

    res: Dict[str, Any] = {
        "runtime": {"fig": components(fig_runtime), "title": "Pipeline Runtime"},
        "performance_1": {"fig": components(performance_1), "title": ""},
        "performance_2": {"fig": components(performance_2), "title": ""},
        "cost": {"fig": components(fig_cost), "title": "Pipeline Cost"},
        "has_error": has_error,
        "error": {"fig": error_result, "title": "Pipeline Error"},
        "codes": comp_res["codes"],
        "system_prompt": comp_res["system_prompt"],
        "usr_prompt": comp_res["usr_prompt"],
        "prompt_tree": render_prompt_tree(system_prompt=comp_res["system_prompt"], user_prompt=comp_res["usr_prompt"])
    }

    return res


class nodeNameChildren:
    def __init__(self, name: str, children):
        self.name = name
        self.children = children

    def add_child(self, node):
        self.children.append(node)

    def to_dict(self):
        node_dict = {'name': self.name}
        if self.children is not None and len(self.children) > 0:
            node_dict['children'] = self.get_children()

        return node_dict

    def get_children(self) -> list | None:
        result = []
        for c in self.children:
            result.append(c.to_dict())
        return result


def get_text_as_list(text: str) -> list:
    result = []
    for line in text.split("\n"):
        result.append(line)
    return result


def render_prompt_tree(system_prompt: str, user_prompt: str):
    sys_list = get_text_as_list(system_prompt)
    usr_list = get_text_as_list(user_prompt)

    sys_children = []

    node_sys_prompt = nodeNameChildren(name="System Prompt", children=[])
    node_usr_prompt = nodeNameChildren(name="Catalog Data", children=[])
    node_sys_rules = nodeNameChildren(name="Rules", children=[])

    tree_data = nodeNameChildren(name="Prompt", children=[node_sys_prompt, node_usr_prompt])
    for sl in sys_list:
        if sl.startswith("###  Task:"):
            task_rule = sl.split(":")[-1]
            tmp_node = nodeNameChildren(name="Task", children=[nodeNameChildren(name=task_rule, children=None)])
            node_sys_prompt.add_child(tmp_node)

        if sl.startswith("###  Input:"):
            input_rule = sl.split(":")[-1]
            tmp_node = nodeNameChildren(name="Description", children=[nodeNameChildren(name=input_rule, children=None)])
            node_sys_prompt.add_child(tmp_node)
            break

    for sl in sys_list:
        for i in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21}:
            if sl.startswith(f"# {i}:"):
                task_rule = sl[len(f"# {i}:"):len(sl)]
                tmp_node = nodeNameChildren(name=task_rule, children=None)
                node_sys_rules.add_child(tmp_node)
                break

    if system_prompt.find('print(f"Train_AUC:{{Train_AUC}}")'):
        tmp_node = nodeNameChildren(name="Evaluation", children=[nodeNameChildren(name="Report AUC ", children=None),
                                                                 nodeNameChildren(name="Report Accuracy",
                                                                                  children=None),
                                                                 nodeNameChildren(name="Report F1-score",
                                                                                  children=None),
                                                                 ])
        node_sys_rules.add_child(tmp_node)

    catalog_data_list = []
    schema_flag = False
    schema_begin = False
    for ul in usr_list:
        if ul.startswith("### Schema"):
            schema_flag = True
        if schema_flag and ul.startswith('"""'):
            schema_begin = True
            schema_flag = False
            continue

        if schema_begin and not ul.startswith('"""'):
            catalog_data_list.append(ul)

        if schema_begin and ul.startswith('"""'):
            break

    categorical_node = nodeNameChildren(name="Categorical Features", children=[])
    numerical_node = nodeNameChildren(name="Numerical Features", children=[])
    other_node = nodeNameChildren(name="String Features", children=[])
    for cd in catalog_data_list:
        if "**This is a target column**" in cd:
            cd = cd.replace(", **This is a target column**", "")
        if "categorical-values" in cd:
            col_name = cd.split(":")[0]
            col_name = col_name.split('" (')[0]
            col_name = col_name.replace('# "', '')

            vals = cd.split("categorical-values [")[-1]
            vals = vals[0: len(vals)-1]
            vals = vals.split(",")
            chl = []
            for v in vals:
                chl.append(nodeNameChildren(name=v, children=None))
            cn_node = nodeNameChildren(name=col_name, children=chl)
            categorical_node.add_child(cn_node)

        elif "min-value" in cd:
            cd = cd.replace("[","").replace("]","").replace("(","").replace(")","")
            str_template = r'# "(.*)" (.*), distinct-count (.*), min-value (.*), max-value (.*), median-value (.*), mean-value (.*)'
            m = re.match(str_template, cd)
            col_name = m.group(1)
            distinct_count = m.group(3)
            min_value = m.group(4)
            max_value = m.group(5)
            mean_value = m.group(6)
            chl = [nodeNameChildren(name=f"Distinct Count: {distinct_count}", children=None),
                   nodeNameChildren(name=f"Min Value: {min_value}", children=None),
                   nodeNameChildren(name=f"Max Value: {max_value}", children=None),
                   nodeNameChildren(name=f"Mean Value: {mean_value}", children=None),]

            cn_node = nodeNameChildren(name=col_name, children=chl)
            numerical_node.add_child(cn_node)
        else:
            cd = cd.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
            str_template = r'# "(.*)" (.*), distinct-count (.*)'
            m = re.match(str_template, cd)
            col_name = m.group(1)
            distinct_count = m.group(3)
            chl = [nodeNameChildren(name=f"Distinct Count: {distinct_count}", children=None)]

            cn_node = nodeNameChildren(name=col_name, children=chl)
            other_node.add_child(cn_node)



    node_sys_prompt.add_child(node_sys_rules)
    node_usr_prompt.add_child(categorical_node)
    node_usr_prompt.add_child(numerical_node)
    node_usr_prompt.add_child(other_node)

    tree_data_str = json.dumps(tree_data.to_dict(), indent=4)
    return "<script>" + tree_data_str + "</script>"


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


def render_bar_runtime_chart(max_time: int, df, yscale: str, plot_width: int, plot_height: int, ylabel: str,
                             xlabel: str) -> figure:
    colors = ["#FFCC85FF", CATEGORY20[0], "red"]
    fig = figure(
        x_range=list(df.index),
        # y_range=[0, max_time],
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
                (f"{val} time", "@$name"),
            ],
            formatters={"@{%s}" % rend[i].name: formatter},
            renderers=[rend[i]],
        )
        fig.add_tools(hover)

    fig.legend.orientation = "horizontal"
    fig.yaxis.axis_label = ylabel
    fig.xaxis.axis_label = xlabel
    tweak_figure(fig)
    relocate_legend(fig, "above")
    return fig


def render_bar_cost_chart(max_cost: int, df, yscale: str, plot_width: int, plot_height: int, ylabel: str,
                          xlabel: str) -> figure:
    colors = [CATEGORY20[0], "red"]
    fig = figure(
        x_range=list(df.index),
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
        width=0.8,
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
    fig.legend.orientation = "horizontal"
    fig.yaxis.axis_label = ylabel
    fig.xaxis.axis_label = xlabel
    tweak_figure(fig)
    relocate_legend(fig, "above")

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
        title: str = '',
        ratio: int = -1
) -> figure:
    width = 0.5
    df["x0"], df["x1"] = df.index + 0.2, df.index + 0.8
    if ratio == -1:
        ratio = df.max(numeric_only=True).max()
    fig = figure(
        width=plot_width,
        height=plot_height,
        title=title,
        toolbar_location=None,
        x_range=df["grp"],
        y_range=[0, ratio],
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


def bar_viz(df: pd.DataFrame, ttl_grps: int, col: str, plot_width: int, bar_width: float, plot_height: int,
            show_yticks: bool,
            bar_cfg: Bar, tooltip_title: str, title: str) -> figure:
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
        y_range=[0, max(df[col])],
    )
    fig.vbar(
        x="index",
        width=bar_width,
        top=col,
        fill_color=bar_cfg.color,
        line_color=bar_cfg.color,
        # bottom=0,
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
