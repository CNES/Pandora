# pylint: skip-file
def compute_matching_costs(img_left, imgs_right, cv, disps, census_width, census_height):
    """
    Given a left image and right images (multiple when doing subpixellic), compute the Census matching costs for all disparities, with the given window.

    :param img_left: the left image
    :type img_left: 2D np.array (row, col) dtype = np.float32
    :param imgs_right: the right images
    :type imgs_right: List of 2D np.array (row, col) dtype = np.float32
    :param cv: cost volume to fill
    :type cv: 3D np.array (row, col, disps) dtype = np.float32
    :param disps: the disparities to sample, sorted
    :type disps: np.array (disps) dtype = np.float32
    :param census_width: the width of the census window
    :type census_width: int
    :param census_height: the height of the census window
    :type census_height: int
    :return: the filled cost volume
    :rtype: 3D np.array (row, col, disps) dtype = np.float32
    """
    ...
