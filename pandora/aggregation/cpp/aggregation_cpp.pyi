# pylint: skip-file
def cbca_step_1(input):
    """
    Given the matching cost for one disparity, build a horizontal integral image storing the cumulative row sum,
    S_h(row, col) = S_h(row-1, col) + cv(row, col)

    :param cv: cost volume for the current disparity
    :type cv: 2D np.array (row, col) dtype = np.float32
    :return: the horizontal integral image, step 1
    :rtype: 2D np.array (row, col + 1) dtype = np.float32
    """
    ...

def cbca_step_2(step1, cross_left, cross_right, range_col, range_col_right):
    """
    Given the horizontal integral image, computed the horizontal matching cost for one disparity,
    E_h(row, col) = S_h(row + right_arm_length, col) - S_h(row - left_arm_length -1, col)

    :param step1: horizontal integral image, from the cbca_step1, with an extra column that contains 0
    :type step1: 2D np.array (row, col + 1) dtype = np.float32
    :param cross_left: cross support of the left image
    :type cross_left: 3D np.array (row, col, [left, right, top, bot]) dtype=np.int16
    :param cross_right: cross support of the right image
    :type cross_right: 3D np.array (row, col, [left, right, tpo, bot]) dtype=np.int16
    :param range_col: left column for the current disparity (i.e : np.arrange(nb columns), where the correspondent \
    in the right image is reachable)
    :type range_col: 1D np.array
    :param range_col_right: right column for the current disparity (i.e : np.arrange(nb columns) - disparity, where \
    column - disparity >= 0 and <= nb columns)
    :type range_col_right: 1D np.array
    :return: the horizontal matching cost for the current disparity, and the number of support pixels used for the \
    step 2
    :rtype: tuple (2D np.array (row, col) dtype = np.float32, 2D np.array (row, col) dtype = np.float32)
    """
    ...

def cbca_step_3(step2):
    """
    Given the horizontal matching cost, build a vertical integral image for one disparity,
    S_v = S_v(row, col - 1) + E_h(row, col)

    :param step2: horizontal matching cost, from the cbca_step2
    :type step2: 3D xarray.DataArray (row, col, disp)
    :return: the vertical integral image for the current disparity
    :rtype: 2D np.array (row + 1, col) dtype = np.float32
    """
    ...

def cbca_step_4(step3, sum2, cross_left, cross_right, range_col, range_col_right):
    """
    Given the vertical integral image, build the fully aggregated matching cost for one disparity,
    E = S_v(row, col + bottom_arm_length) - S_v(row, col - top_arm_length - 1)

    :param step3: vertical integral image, from the cbca_step3, with an extra row that contains 0
    :type step3: 2D np.array (row + 1, col) dtype = np.float32
    :param sum2: the number of support pixels used for the step 2
    :type sum2: 2D np.array (row, col) dtype = np.float32
    :param cross_left: cross support of the left image
    :type cross_left: 3D np.array (row, col, [left, right, top, bot]) dtype=np.int16
    :param cross_right: cross support of the right image
    :type cross_right: 3D np.array (row, col, [left, right, tpo, bot]) dtype=np.int16
    :param range_col: left column for the current disparity (i.e : np.arrange(nb columns), where the correspondent \
    in the right image is reachable)
    :type range_col: 1D np.array
    :param range_col_right: right column for the current disparity (i.e : np.arrange(nb columns) - disparity, where \
    column - disparity >= 0 and <= nb columns)
    :type range_col_right: 1D np.array
    :return: the fully aggregated matching cost, and the total number of support pixels used for the aggregation
    :rtype: tuple(2D np.array (row , col) dtype = np.float32, 2D np.array (row , col) dtype = np.float32)
    """
    ...

def cross_support(image, len_arms, intensity):
    """
    Compute the cross support for an image: find the 4 arms.
    Enforces a minimum support region of 3×3 if pixels are valid.
    The cross support of invalid pixels (pixels that are np.inf) is 0 for the 4 arms.

    :param image: image
    :type image: 2D np.array (row , col) dtype = np.float32
    :param len_arms: maximal length arms
    :param len_arms: int16
    :param intensity: maximal intensity
    :param intensity: float 32
    :return: a 3D np.array ( row, col, [left, right, top, bot] ), with the four arms lengths computes for each pixel
    :rtype:  3D np.array ( row, col, [left, right, top, bot] ), dtype=np.int16
    """
    ...
