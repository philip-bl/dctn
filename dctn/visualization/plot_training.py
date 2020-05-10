import os.path
from typing import Tuple, Optional

from bokeh.models import ColumnDataSource, Range1d, Slider, RangeSlider
from bokeh.layouts import gridplot
from bokeh.plotting import save, output_file, figure, Figure


from libcrap.visualization import get_distinguishable_colors

from dctn.visualization.log_parsing import Record, load_records

base_dir = "/mnt/important/experiments"
log_rel_fname = "log.log"

experiments_rel_dirs: Tuple[str, ...] = (
    "eps_plus_linear_fashionmnist/replicate_90.19_vacc/2020-05-05T19:45:54_stopped_manually",
    "eps_plus_linear_fashionmnist/2020-04-28T19:52:08",
)
experiments_names = (
    "Very small stds",
    "Init to make intermediate xs have std=1 empirically",
)
assert len(experiments_names) == len(experiments_rel_dirs)
colors = get_distinguishable_colors(len(experiments_names))


all_increasing_tracc_records: Tuple[Tuple[Record, ...], ...] = tuple(
    load_records(os.path.join(base_dir, experiment_dir, log_rel_fname), increasing_tracc=True)
    for experiment_dir in experiments_rel_dirs
)

all_records: Tuple[Tuple[Record, ...], ...] = tuple(
    load_records(os.path.join(base_dir, experiment_dir, log_rel_fname), increasing_tracc=False)
    for experiment_dir in experiments_rel_dirs
)


output_file("one_eps_vacc_by_tracc.html")

tools = "pan,wheel_zoom,box_zoom,reset,crosshair,hover"

tracc_range = Range1d(bounds=(0.0, 1.0))
vacc_range = Range1d(bounds=(0.0, 1.0))
nitd_range = Range1d(
    0,
    (maximum_nitd := max(records[-1].nitd for records in all_records)),
    bounds=(0, maximum_nitd),
)
min_mce = min(
    min(min(record.trmce for record in records) for records in all_records),
    min(min(record.vmce for record in records) for records in all_records),
)
max_mce = max(
    max(max(record.trmce for record in records) for records in all_records),
    max(max(record.vmce for record in records) for records in all_records),
)
trmce_range = Range1d(0.0, max_mce, bounds=(min_mce, max_mce))
vmce_range = Range1d(0.0, max_mce, bounds=(min_mce, max_mce))

# plot vacc by tracc
vacc_by_tracc_plot = figure(
    title="One EPS + linear",
    x_axis_label="train acc",
    y_axis_label="val acc",
    tools=tools,
    x_range=tracc_range,
    y_range=vacc_range,
)
vacc_by_tracc_plot.line(
    (0.0, 1.0), (0.0, 1.0), line_color="black", alpha=0.3, line_dash="dashed"
)
for experiment_name, records, color in zip(
    experiments_names, all_increasing_tracc_records, colors
):
    vacc_by_tracc_plot.line(
        tuple(record.tracc for record in records),
        tuple(record.vacc for record in records),
        legend_label=experiment_name,
        line_color=color,
    )
vacc_by_tracc_plot.legend.location = "top_left"
vacc_by_tracc_plot.legend.click_policy = "hide"


def plot_something_by_nitd(
    y_axis_label: str,
    y_range: Range1d,
    record_attr: str,
    legend_location: str,
    plot_height: Optional[int] = None,
) -> Figure:
    plot = figure(
        x_axis_label="number of iterations done",
        y_axis_label=y_axis_label,
        tools=tools,
        x_range=nitd_range,
        y_range=y_range,
        plot_height=plot_height,
    )
    for experiment_name, records, color in zip(experiments_names, all_records, colors):
        plot.line(
            tuple(record.nitd for record in records),
            tuple(getattr(record, record_attr) for record in records),
            legend_label=experiment_name,
            line_color=color,
        )
    plot.legend.location = legend_location
    plot.legend.click_policy = "hide"
    return plot


x_by_nitd_plot_height = 300
vacc_by_nitd_plot = plot_something_by_nitd(
    "val acc", vacc_range, "vacc", "bottom_right", x_by_nitd_plot_height
)
tracc_by_nitd_plot = plot_something_by_nitd(
    "train acc", tracc_range, "tracc", "bottom_right", x_by_nitd_plot_height
)
vmce_by_nitd_plot = plot_something_by_nitd(
    "val mean negative log likelihood", vmce_range, "vmce", "top_right", x_by_nitd_plot_height
)
trmce_by_nitd_plot = plot_something_by_nitd(
    "train mean negative log likelihood",
    trmce_range,
    "trmce",
    "top_right",
    x_by_nitd_plot_height,
)


def create_range_slider(range: Range1d, title: str, step: float) -> RangeSlider:
    slider = RangeSlider(
        start=range.start,
        end=range.end,
        step=step,
        value=(range.bounds[0], range.bounds[1]),
        title=title,
    )
    slider.js_link("value", range, "start", attr_selector=0)
    slider.js_link("value", range, "end", attr_selector=1)
    return slider


vmce_slider = create_range_slider(vmce_range, "val mean negative log likelihood", 0.05)
trmce_slider = create_range_slider(trmce_range, "train mean negative log likelihood", 0.05)
vacc_slider = create_range_slider(vacc_range, "val acc", 0.005)
tracc_slider = create_range_slider(tracc_range, "train acc", 0.005)
nitd_slider = create_range_slider(nitd_range, "number of iterations done", 50)

p = gridplot(
    (
        (vacc_by_tracc_plot,),
        (vacc_slider, tracc_slider),
        (vacc_by_nitd_plot, tracc_by_nitd_plot),
        (vmce_slider, trmce_slider),
        (nitd_slider,),
        (vmce_by_nitd_plot, trmce_by_nitd_plot),
    )
)

save(p)

# TODO:
# better hover tooltips
# legend outside of the figure
# being able to disable an experiment
# toolbar for each of the plots
# Add tooltips in the legend maybe
# fix not being able to scroll when one of the ranges hit the bound
