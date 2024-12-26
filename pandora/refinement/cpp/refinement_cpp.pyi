# pylint: skip-file
def quadratic_refinement_method(cost, disp, measure, cst_pandora_msk_pixel_stopped_interpolation): ...
def loop_refinement(
    cv,
    disp,
    mask,
    d_min,
    d_max,
    subpixel,
    measure,
    method,
    cst_pandora_msk_pixel_invalid,
    cst_pandora_msk_pixel_stopped_interpolation,
): ...
def loop_approximate_refinement(
    cv,
    disp,
    mask,
    d_min,
    d_max,
    subpixel,
    measure,
    method,
    cst_pandora_msk_pixel_invalid,
    cst_pandora_msk_pixel_stopped_interpolation,
): ...
def vfit_refinement_method(cost, disp, measure, cst_pandora_msk_pixel_stopped_interpolation): ...
