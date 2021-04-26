# pylint: skip-file

import copy
import warnings
from typing import Tuple

import numpy as np
import xarray as xr
import os

from pandora.constants import *
from pandora.img_tools import rasterio_open

warnings.simplefilter(action="ignore")
from bokeh.colors import RGB
import bokeh.plotting as bpl
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, Legend
from bokeh.io import show, output_notebook
import ipyvolume as ipv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.image import imsave


def pandora_cmap():

    colors = ["crimson", "lightpink", "white", "yellowgreen"]
    nodes = [0.0, 0.4, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    return cmap_shift


def cmap_to_palette(cmap):

    cmap_rgb = (255 * cmap(range(256))).astype("int")
    palette = [RGB(*tuple(rgb)).to_hex() for rgb in cmap_rgb]

    return palette


def plot_disparity(input_disparity_map: xr.Dataset) -> None:
    """
    Plot disparity map with selective bit mask
    :param input_disparity_map: input disparity map
    :type  input_disparity_map: xr.dataset
    :return: None
    """

    output_notebook()
    disparity_map = add_validity_mask_to_dataset(input_disparity_map)
    valid_idx = np.where(disparity_map["validity_mask"].data == 0)
    min_d = np.nanmin(disparity_map["disparity_map"].data[valid_idx])
    max_d = np.nanmax(disparity_map["disparity_map"].data[valid_idx])
    cmap_pandora = pandora_cmap()
    mapper_avec_opti = LinearColorMapper(palette=cmap_to_palette(cmap_pandora), low=min_d, high=max_d)

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0, 0)
    )
    size = 0.5
    dw = disparity_map["disparity_map"].shape[1]
    dh = disparity_map["disparity_map"].shape[0]

    fig = figure(
        title="Disparity map", width=800, height=450, tools=["reset", "pan", "box_zoom"], output_backend="webgl"
    )

    fig.image(
        image=[np.flip(disparity_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    # Only add to the legend the masks that are not empty
    legend_items = []

    x = np.where(disparity_map["nodata_border_left_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["nodata_border_left_mask"].data != 0)[0]
    nodata_border_left_mask = fig.circle(x=x, y=y, size=size, color="black")
    nodata_border_left_mask.visible = False
    if x != []:
        legend_items.append(("Nodata border left_mask (invalid)", [nodata_border_left_mask]))

    x = np.where(disparity_map["nodata_border_right_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["nodata_border_right_mask"].data != 0)[0]
    nodata_border_right_mask = fig.circle(x=x, y=y, size=size, color="black")
    nodata_border_right_mask.visible = False
    if x != []:
        legend_items.append(("Nodata border right_mask (invalid)", [nodata_border_right_mask]))

    x = np.where(disparity_map["incomplete_right_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["incomplete_right_mask"].data != 0)[0]
    incomplete_right_mask = fig.circle(x=x, y=y, size=size, color="black")
    incomplete_right_mask.visible = False
    if x != []:
        legend_items.append(("Incomplete right mask", [incomplete_right_mask]))

    x = np.where(disparity_map["stopped_interp_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["stopped_interp_mask"].data != 0)[0]
    stopped_interp_mask = fig.circle(x=x, y=y, size=size, color="black")
    stopped_interp_mask.visible = False
    if x != []:
        legend_items.append(("Stopped interp mask", [stopped_interp_mask]))

    x = np.where(disparity_map["filled_occlusion_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["filled_occlusion_mask"].data != 0)[0]
    filled_occlusion_mask = fig.circle(x=x, y=y, size=size, color="black")
    filled_occlusion_mask.visible = False
    if x != []:
        legend_items.append(("Filled occlusion mask", [filled_occlusion_mask]))

    x = np.where(disparity_map["filled_mismatch_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["filled_mismatch_mask"].data != 0)[0]
    filled_mismatch_mask = fig.circle(x=x, y=y, size=size, color="black")
    filled_mismatch_mask.visible = False
    if x != []:
        legend_items.append(("Filled mismatch mask", [filled_mismatch_mask]))

    x = np.where(disparity_map["masked_left_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["masked_left_mask"].data != 0)[0]
    masked_left_mask = fig.circle(x=x, y=y, size=size, color="black")
    masked_left_mask.visible = False
    if x != []:
        legend_items.append(("Masked left mask (invalid)", [masked_left_mask]))

    x = np.where(disparity_map["masked_right_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["masked_right_mask"].data != 0)[0]
    masked_right_mask = fig.circle(x=x, y=y, size=size, color="black")
    masked_right_mask.visible = False
    if x != []:
        legend_items.append(("Masked right mask (invalid)", [masked_right_mask]))

    x = np.where(disparity_map["occlusion_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["occlusion_mask"].data != 0)[0]
    occlusion_mask = fig.circle(x=x, y=y, size=size, color="black")
    occlusion_mask.visible = False
    if x != []:
        legend_items.append(("Occlusion mask (invalid)", [occlusion_mask]))

    x = np.where(disparity_map["mismatch_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["mismatch_mask"].data != 0)[0]
    mismatch_mask = fig.circle(x=x, y=y, size=size, color="black")
    mismatch_mask.visible = False
    if x != []:
        legend_items.append(("Mismatch mask (invalid)", [mismatch_mask]))

    x = np.where(disparity_map["filled_nodata"].data != 0)[1]
    y = dh - np.where(disparity_map["filled_nodata"].data != 0)[0]
    filled_nodata = fig.circle(x=x, y=y, size=size, color="black")
    filled_nodata.visible = False
    if x != []:
        legend_items.append(("Filled nodata", [filled_nodata]))

    x = np.where(disparity_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(disparity_map["invalid_mask"].data != 0)[0]
    invalid_mask = fig.circle(x=x, y=y, size=size, color="black")
    invalid_mask.visible = True
    if x != []:
        legend_items.append(("All invalid types", [invalid_mask]))

    legend = Legend(items=legend_items, location="center", click_policy="hide")

    fig.add_layout(color_bar, "right")
    fig.add_layout(legend, "right")

    show(fig)


def adapt_occlusion_mask(mask_path: str, output_dir: str, valid_value: int = None, title: str = None) -> str:
    """
    Adapt occlusion mask to Pandora's standard, where 0 is valid and >1 invalid
    :param mask_path: path to input occlusion mask
    :type mask_path: str
    :param output_dir: directory to save adapted mask
    :type output_dir: str
    :param valid_value: known mask's valid value
    :type valid_value: int
    :param title: title of the adapted mask image to save
    :type title: str
    :return: output image path
    :rtype: str
    """
    mask = rasterio_open(mask_path).read(1)
    # If no valid value was set, the lowest value of the mask is considered the valid one
    if valid_value == None:
        valid_value = np.nanmin(mask)
    # Initialize and fill new mask
    output_mask = np.zeros(mask.shape)
    inv_idx = np.where(mask != valid_value)
    output_mask[inv_idx] = 1
    imsave(os.path.join(output_dir, title + ".png"), output_mask.astype(np.uint8), cmap=cm.gray)
    return os.path.join(output_dir, title + ".png")


def compare_2_disparities(
    input_first_disp_map: xr.Dataset, first_title: str, input_second_disp_map: xr.Dataset, second_title: str
) -> None:
    """
    Show 2 disparity maps
    :param input_first_disp_map: disparity map
    :type input_first_disp_map: dataset
    :param first_title: disparity map title
    :type first_title: str
    :param input_second_disp_map: disparity map
    :type input_second_disp_map: dataset
    :param second_title: disparity map title
    :type second_title: str
    :return: none
    """
    output_notebook()
    size = 0.5

    first_disp_map = add_validity_mask_to_dataset(input_first_disp_map)
    second_disp_map = add_validity_mask_to_dataset(input_second_disp_map)

    valid_idx = np.where(first_disp_map["validity_mask"].data == 0)
    min_d = np.nanmin(first_disp_map["disparity_map"].data[valid_idx])
    max_d = np.nanmax(first_disp_map["disparity_map"].data[valid_idx])

    cmap_pandora = pandora_cmap()
    mapper_avec_opti = LinearColorMapper(palette=cmap_to_palette(cmap_pandora), low=min_d, high=max_d)

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0, 0)
    )

    dw = first_disp_map["disparity_map"].shape[1]
    dh = first_disp_map["disparity_map"].shape[0]

    if first_title == None:
        first_title = "First disparity map"
    if second_title == None:
        second_title = "Second disparity map"

    # First disparity map
    first_fig = figure(
        title=first_title, width=450, height=450, tools=["reset", "pan", "box_zoom"], output_backend="webgl"
    )

    first_fig.image(
        image=[np.flip(first_disp_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    x = np.where(first_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(first_disp_map["invalid_mask"].data != 0)[0]
    first_inv_msk = first_fig.circle(x=x, y=y, size=size, color="black")

    legend = Legend(items=[("inv msk", [first_inv_msk])], location="center", click_policy="hide")

    first_fig.add_layout(legend, "below")
    first_fig.add_layout(color_bar, "right")

    # Second disparity map
    second_fig = figure(
        title=second_title,
        width=450,
        height=450,
        tools=["reset", "pan", "box_zoom"],
        output_backend="webgl",
        x_range=first_fig.x_range,
        y_range=first_fig.y_range,
    )

    second_fig.image(
        image=[np.flip(second_disp_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    x = np.where(second_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(second_disp_map["invalid_mask"].data != 0)[0]
    second_inv_msk = second_fig.circle(x=x, y=y, size=size, color="black")

    legend = Legend(
        items=[("inv msk", [second_inv_msk])], glyph_height=10, glyph_width=10, location="center", click_policy="hide"
    )

    second_fig.add_layout(legend, "below")
    second_fig.add_layout(color_bar, "right")

    layout = column(row(first_fig, second_fig))

    show(layout)


def compare_3_disparities_and_error(
    input_first_disp_map: xr.Dataset,
    first_title: str,
    input_second_disp_map: xr.Dataset,
    second_title: str,
    input_third_disp_map: xr.Dataset,
    third_title: str,
    error_map: np.array,
    error_title: str,
) -> None:
    """
    Show 3 disparity maps and error
    :param input_first_disp_map: disparity map
    :type input_first_disp_map: dataset
    :param first_title: disparity map title
    :type first_title: str
    :param input_second_disp_map: disparity map
    :type input_second_disp_map: dataset
    :param second_title: disparity map title
    :type second_title: str
    :param input_third_disp_map: disparity map
    :type input_third_disp_map: dataset
    :param third_title: disparity map title
    :type third_title: str
    :param error_map: error map
    :type error_map: np.array
    :param error_title: error title
    :type error_title: str
    :return: none
    """
    output_notebook()
    size = 0.5

    first_disp_map = add_validity_mask_to_dataset(input_first_disp_map)
    second_disp_map = add_validity_mask_to_dataset(input_second_disp_map)
    third_disp_map = add_validity_mask_to_dataset(input_third_disp_map)

    valid_idx = np.where(first_disp_map["validity_mask"].data == 0)
    min_d = np.nanmin(first_disp_map["disparity_map"].data[valid_idx])
    max_d = np.nanmax(first_disp_map["disparity_map"].data[valid_idx])
    cmap_pandora = pandora_cmap()
    mapper_avec_opti = LinearColorMapper(palette=cmap_to_palette(cmap_pandora), low=min_d, high=max_d)

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0, 0)
    )

    dw = first_disp_map["disparity_map"].shape[1]
    dh = first_disp_map["disparity_map"].shape[0]

    # First disparity map
    first_fig = figure(
        title=first_title, width=400, height=400, tools=["reset", "pan", "box_zoom"], output_backend="webgl"
    )

    first_fig.image(
        image=[np.flip(first_disp_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    x = np.where(first_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(first_disp_map["invalid_mask"].data != 0)[0]
    first_inv_msk = first_fig.circle(x=x, y=y, size=size, color="black")
    legend = Legend(items=[("inv msk", [first_inv_msk])], location="center", click_policy="hide")
    first_fig.add_layout(legend, "below")
    first_fig.add_layout(color_bar, "right")

    # Second disparity map
    second_fig = figure(
        title=second_title,
        width=400,
        height=400,
        tools=["reset", "pan", "box_zoom"],
        output_backend="webgl",
        x_range=first_fig.x_range,
        y_range=first_fig.y_range,
    )

    second_fig.image(
        image=[np.flip(second_disp_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    x = np.where(second_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(second_disp_map["invalid_mask"].data != 0)[0]
    second_inv_msk = second_fig.circle(x=x, y=y, size=size, color="black")
    legend = Legend(
        items=[("inv msk", [second_inv_msk])], glyph_height=10, glyph_width=10, location="center", click_policy="hide"
    )
    second_fig.add_layout(legend, "below")
    second_fig.add_layout(color_bar, "right")

    # Third disparity map
    third_fig = figure(
        title=third_title,
        width=400,
        height=400,
        tools=["reset", "pan", "box_zoom"],
        output_backend="webgl",
        x_range=first_fig.x_range,
        y_range=first_fig.y_range,
    )

    third_fig.image(
        image=[np.flip(third_disp_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    x = np.where(third_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(third_disp_map["invalid_mask"].data != 0)[0]
    third_inv_msk = third_fig.circle(x=x, y=y, size=size, color="black")

    legend = Legend(
        items=[("inv msk", [third_inv_msk])], glyph_height=10, glyph_width=10, location="center", click_policy="hide"
    )
    third_fig.add_layout(legend, "below")
    third_fig.add_layout(color_bar, "right")

    # Error plot
    error_fig = figure(
        title=error_title,
        width=400,
        height=400,
        tools=["reset", "pan", "box_zoom"],
        output_backend="webgl",
        x_range=first_fig.x_range,
        y_range=first_fig.y_range,
    )
    min_d = 0
    max_d = 10
    reds_cmap = cm.get_cmap("Reds", 256)
    mapper_avec_opti = LinearColorMapper(palette=cmap_to_palette(reds_cmap), low=min_d, high=max_d)

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0, 0)
    )
    error_fig.image(image=[np.flip(error_map, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti)
    error_fig.add_layout(color_bar, "right")

    layout = column(row(first_fig, second_fig), row(third_fig, error_fig))

    show(layout)


def compare_disparity_and_error(
    input_first_disp_map: xr.Dataset, first_title: str, error_map: np.array, error_title: str
) -> None:
    """
    Show disparity map and error
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param title: disparity map title
    :type title: str
    :param error_map: error map
    :type error_map: np.array
    :param error_title: error title
    :type error_title: str
    :return: none
    """
    output_notebook()
    size = 0.5

    # Disparity map
    first_disp_map = add_validity_mask_to_dataset(input_first_disp_map)

    valid_idx = np.where(first_disp_map["validity_mask"].data == 0)
    min_d = np.nanmin(first_disp_map["disparity_map"].data[valid_idx])
    max_d = np.nanmax(first_disp_map["disparity_map"].data[valid_idx])
    cmap_pandora = pandora_cmap()
    mapper_avec_opti = LinearColorMapper(palette=cmap_to_palette(cmap_pandora), low=min_d, high=max_d)

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0, 0)
    )

    dw = first_disp_map["disparity_map"].shape[1]
    dh = first_disp_map["disparity_map"].shape[0]

    first_fig = figure(
        title=first_title,
        width=450,
        height=450,
        tools=["reset", "pan", "box_zoom"],
        output_backend="webgl",
        sizing_mode="scale_width",
    )
    first_fig.image(
        image=[np.flip(first_disp_map["disparity_map"].data, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti
    )

    x = np.where(first_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(first_disp_map["invalid_mask"].data != 0)[0]
    first_inv_msk = first_fig.circle(x=x, y=y, size=size, color="black")
    legend = Legend(items=[("inv msk", [first_inv_msk])], location="center", click_policy="hide")

    first_fig.add_layout(legend, "below")
    first_fig.add_layout(color_bar, "right")

    # Error plot
    error_fig = figure(
        title=error_title,
        width=450,
        height=450,
        tools=["reset", "pan", "box_zoom"],
        output_backend="webgl",
        x_range=first_fig.x_range,
        y_range=first_fig.y_range,
        sizing_mode="scale_width",
    )
    min_d = 0
    max_d = 10
    reds_cmap = cm.get_cmap("Reds", 256)
    mapper_avec_opti = LinearColorMapper(palette=cmap_to_palette(reds_cmap), low=min_d, high=max_d)

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti, ticker=BasicTicker(), label_standoff=12, border_line_color=None, location=(0, 0)
    )
    error_fig.image(image=[np.flip(error_map, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti)

    # Add disparity's error mask on error
    x = np.where(first_disp_map["invalid_mask"].data != 0)[1]
    y = dh - np.where(first_disp_map["invalid_mask"].data != 0)[0]
    error_inv_msk = error_fig.circle(x=x, y=y, size=size, color="black")
    legend = Legend(items=[("inv msk", [error_inv_msk])], location="center", click_policy="hide")

    error_fig.add_layout(legend, "below")
    error_fig.add_layout(color_bar, "right")

    layout = column(row(first_fig, error_fig))
    show(layout)


def show_input_images(img_left: xr.Dataset, img_right: xr.Dataset) -> None:
    """
    Show input images and anaglyph
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param right_disp_map: right disparity map
    :type right_disp_map: dataset
    :return: none
    """
    output_notebook()

    dw = np.flip(img_left.im.data, 0).shape[1]
    dh = np.flip(img_right.im.data, 0).shape[0]
    width = 320
    height = 320

    # Image left
    img_left_data = img_left.im.data
    left_fig = bpl.figure(title="Left image", width=width, height=height)
    left_fig.image(image=[np.flip(img_left_data, 0)], x=1, y=0, dw=dw, dh=dh)

    # Image right
    img_right_data = img_right.im.data
    right_fig = bpl.figure(
        title="Right image", width=width, height=height, x_range=left_fig.x_range, y_range=left_fig.y_range
    )
    right_fig.image(image=[np.flip(img_right_data, 0)], x=1, y=0, dw=dw, dh=dh)

    # Anaglyph
    img_left, img_right_align = xr.align(img_left, img_right)
    anaglyph = np.stack((img_left.im, img_right_align.im, img_right_align.im), axis=-1)

    clip_percent = 5
    vmin_ref = np.percentile(img_left.im, clip_percent)
    vmax_ref = np.percentile(img_left.im, 100 - clip_percent)
    vmin_sec = np.percentile(img_right.im, clip_percent)
    vmax_sec = np.percentile(img_right.im, 100 - clip_percent)
    vmin_anaglyph = np.array([vmin_ref, vmin_sec, vmin_sec])
    vmax_anaglyph = np.array([vmax_ref, vmax_sec, vmax_sec])
    img = np.clip((anaglyph - vmin_anaglyph) / (vmax_anaglyph - vmin_anaglyph), 0, 1)
    anaglyph = np.clip((anaglyph - vmin_anaglyph) / (vmax_anaglyph - vmin_anaglyph), 0, 1)

    # Add 4rth channel to use bokeh's image_rgba function
    img = np.empty((dh, dw), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(dh, dw, 4)
    for i in range(dh):
        for j in range(dw):
            view[i, j, 0] = anaglyph[i, j, 0] * 255
            view[i, j, 1] = anaglyph[i, j, 1] * 255
            view[i, j, 2] = anaglyph[i, j, 2] * 255
            view[i, j, 3] = 255
    anaglyph_fig = bpl.figure(
        title="Anaglyph", width=width, height=height, x_range=left_fig.x_range, y_range=left_fig.y_range
    )
    anaglyph_fig.image_rgba(image=[np.flip(img, 0)], x=1, y=0, dw=dw, dh=dh)
    layout = column(row(left_fig, right_fig, anaglyph_fig))
    show(layout)


def get_error(
    left_disp_map: xr.Dataset, ground_truth: xr.Dataset, threshold: int = 1
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Return error map

    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param ground_truth: ground truth
    :type ground_truth: dataset
    :param threshold: error threshold
    :type threshold: int
    :return: error_map, total_bad_percentage, mean_error, std_error, invalid_percentage
    :rtype: Tuple[np.array,np.array,np.array,np.array,np.array]
    """
    total_bad_percentage, mean_error, std_error, invalid_percentage, error_map = compare_to_gt(
        left_disp_map, ground_truth, threshold, None
    )
    return error_map, total_bad_percentage, mean_error, std_error, invalid_percentage


def plot_1_cost_volume(cv: xr.Dataset, left_disp_map: xr.Dataset, title: str) -> None:
    """
    Plot 3d cost volume

    :param cv: cost volume
    :type cv: dataset
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :return: none
    """
    print(title)
    get_3D_cost_volume(cv, left_disp_map)
    ipv.show()


def get_3D_cost_volume(cv: xr.Dataset, left_disp_map: xr.Dataset) -> None:
    """
    Plot 3d cost volume

    :param cv: cost volume
    :type cv: dataset
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :return: none
    """

    nb_rows, nb_cols, nb_disps = cv["cost_volume"].shape

    X, Y = np.meshgrid(np.arange(nb_cols), np.arange(nb_rows))
    X = np.float32(X)
    Y = np.float32(Y)
    Z = left_disp_map["disparity_map"].data

    color_disp = np.ravel(Z)
    color_disp = color_disp - np.nanmin(color_disp)
    color_disp = color_disp * 1.0 / np.nanmax(color_disp)
    color_disp = np.repeat(color_disp[:, np.newaxis], 3, axis=1)

    fig = ipv.figure()
    scatter = ipv.scatter(np.ravel(X), np.ravel(Y), np.ravel(Z), marker="point_2d", size=10, color=color_disp)
    ipv.ylim(nb_rows, 0)
    ipv.style.box_off()
    ipv.style.use("minimal")

    return ipv.gcc()


def add_mask(all_validity_mask: np.array, msk_type: int) -> np.array:
    """
    Create mask for a given bit

    :param all_validity_mask: mask for all bits
    :type all_validity_mask: np.array
    :return: msk
    :rtype: np.array
    """
    # Mask initialization to 0 (all valid)
    msk = np.full(all_validity_mask.shape, 0)
    # Identify and fill invalid points
    inv_idx = np.where((all_validity_mask & msk_type) != 0)
    msk[inv_idx] = 1
    return msk


def add_validity_mask_to_dataset(input_disp_map: xr.Dataset) -> xr.Dataset:
    """
    Adds validity mask to imput dataset

    :param input_disp_map: disparity map
    :type input_disp_map: dataset
    :return: input_disp_map
    :rtype: dataset
    """

    disp_map = copy.deepcopy(input_disp_map)

    # Invalid
    disp_map["invalid_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_INVALID)), dims=["row", "col"]
    )
    # Bit 0: Edge of the left image or nodata in left image
    disp_map["nodata_border_left_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER)),
        dims=["row", "col"],
    )
    # Bit 1: Disparity interval to explore is missing or nodata in the right image
    disp_map["nodata_border_right_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING)),
        dims=["row", "col"],
    )
    # Bit 2: Incomplete disparity interval in right image
    disp_map["incomplete_right_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE)),
        dims=["row", "col"],
    )
    # Bit 3: Unsuccesful sub-pixel interpolation
    disp_map["stopped_interp_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION)),
        dims=["row", "col"],
    )
    # Bit 4: Filled occlusion
    disp_map["filled_occlusion_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_FILLED_OCCLUSION)), dims=["row", "col"]
    )
    # Bit 5: Filled mismatch
    disp_map["filled_mismatch_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_FILLED_MISMATCH)), dims=["row", "col"]
    )
    # Bit 6: Pixel is masked on the mask of the left image
    disp_map["masked_left_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT)),
        dims=["row", "col"],
    )
    # Bit 7: Disparity to explore is masked on the mask of the right image
    disp_map["masked_right_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT)),
        dims=["row", "col"],
    )
    # Bit 8: Pixel located in an occlusion region
    disp_map["occlusion_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_OCCLUSION)), dims=["row", "col"]
    )
    # Bit 9: Mismatch
    disp_map["mismatch_mask"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_MISMATCH)), dims=["row", "col"]
    )
    # Bit 10: Filled nodata
    disp_map["filled_nodata"] = xr.DataArray(
        np.copy(add_mask(disp_map["validity_mask"].values, PANDORA_MSK_PIXEL_FILLED_NODATA)), dims=["row", "col"]
    )
    return disp_map


def compare_to_gt(
    disp_map: xr.Dataset, ground_truth: xr.Dataset, error_threshold: int, no_data_gt_value: float = None
) -> Tuple[float, float, float, float, np.array]:
    """
    Compute difference between a disparity map (estimated by a stereo tool) and ground_truth.
    Point p is considered as an error if disp_map(p)-ground_truth(p) > threshold
    Statistics (mean, median, standard deviation) are computed regarded error points

    :param disp: disparity map
    :type disp: dataset
    :param ground_truth: ground_truth
    :type ground_truth: dataset
    :param error_threshold: threshold
    :type error_threshold: int
    :param no_data_gt_value: value of ground truth no data
    :type no_data_gt_value: float
    :param invalid_point: True if disparity map contains invalid value (must be NaN)
    :type invalid_point: bool
    :return:
            - total_bad_percentage
            - mean_error
            - std_error
            - map error
    :rtype: float, float, float, 2d numpy array
    """
    disp = disp_map.disparity_map.data
    gt = ground_truth.disparity_map.data
    # Compare Sizes
    if disp.size != gt.size:
        raise ValueError("Ground truth and disparity map must have the same size")

    # Difference between disp_map and ground truth
    error = abs(disp - gt)
    # Do not consider errors lower than the error threshold
    error[np.where(error < error_threshold)] = 0
    # Number of points
    num_points = disp.shape[0] * disp.shape[1]

    # If occlusion mask exists, number of occlusion points is computed.
    # Occlusion points become NaN on error array
    num_occl = 0
    if ground_truth.validity_mask is not None:
        mask = ground_truth.validity_mask
        # Occlusion point value must be different to 0
        occl_coord = np.where(mask != 0)
        num_occl = len(occl_coord[0])
        error[occl_coord] = np.nan
    else:
        mask = np.zeros(ground_truth.disparity_map.shape)

    # All no_data_gt_values become NaN value
    num_no_data_gt = 0
    if no_data_gt_value is not None:
        if no_data_gt_value == np.inf:
            no_data_coord = np.where(np.isinf(gt) & (mask == 0))
        else:
            no_data_coord = np.where((gt == no_data_gt_value) & (mask == 0))
        num_no_data_gt = len(no_data_coord[0])
        error[no_data_coord] = np.nan

    # Invalid point on disparity map
    invalid_coord = np.where(np.isnan(disp) & (mask == 0) & (gt != no_data_gt_value))
    num_invalid = len(invalid_coord[0])
    error[invalid_coord] = np.nan
    # Number of bad points
    bad_coord = np.where(error > 0)
    num_bad = len(bad_coord[0])

    # Percentage of total bad points (bad + invalid)
    total_bad_percentage = ((num_bad + num_invalid) / float(num_points - num_no_data_gt - num_occl)) * 100

    inf_idx = np.where(np.isinf(error))
    error[inf_idx] = np.nan
    # Mean error
    mean_error = float(np.nanmean(error))
    # Standard deviation
    std_error = float(np.nanstd(error))
    # Percentage of invalid points
    invalid_percentage = (num_invalid / float(num_points)) * 100.0

    return total_bad_percentage, mean_error, std_error, invalid_percentage, error
