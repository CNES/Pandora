# pylint: skip-file
def compute_ambiguity(cv, etas, nbr_etas, grids, disparity_range, type_measure_min):
    """
    Computes ambiguity.

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param etas: range between eta_min and eta_max with step eta_step
    :type etas: np.ndarray
    :param nbr_etas: number of etas
    :type nbr_etas: int
    :param grids: array containing min and max disparity grids
    :type grids: 2D np.ndarray (min, max)
    :param disparity_range: array containing disparity range
    :type disparity_range: np.ndarray
    :param type_measure_min: True for min and False for max
    :type type_measure_min: bool
    :return: the normalized ambiguity
    :rtype: 2D np.ndarray (row, col) dtype = float32
    """
    ...

def compute_ambiguity_and_sampled_ambiguity(cv, etas, nbr_etas, grids, disparity_range):
    """
    Return the ambiguity and sampled ambiguity, useful for evaluating ambiguity in notebooks

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param etas: range between eta_min and eta_max with step eta_step
    :type etas: np.ndarray
    :param nbr_etas: nuber of etas
    :type nbr_etas: int
    :param grids: array containing min and max disparity grids
    :type grids: 2D np.ndarray (min, max)
    :param disparity_range: array containing disparity range
    :type disparity_range: np.ndarray
    :return: the normalized ambiguity and sampled ambiguity
    :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 3D np.ndarray (row, col) dtype = float32)
    """
    return None, None

def compute_interval_bounds(cv, disp_interval, possibility_threshold, type_factor):
    """
    Computes interval bounds on the disparity.

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param disp_interval: disparity data
    :type disp_interval: 1D np.ndarray (disp,)
    :param possibility_threshold: possibility threshold used for interval computation
    :type possibility_threshold: float
    :param type_factor: Either 1 or -1. Used to adapt the possibility computation to max or min measures
    :type type_factor: float
    :return: the infimum and supremum (not regularized) of the set containing the true disparity
    :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32)
    """
    return None, None

def compute_risk_and_sampled_risk(cv, sampled_ambiguity, etas, nbr_etas, grids, disparity_range, sample_risk):
    """
    Computes minimum and maximum risk, and sampled_risk if asked to.

    :param cv: cost volume
    :type cv: 3D np.ndarray (row, col, disp)
    :param sampled_ambiguity: sampled cost volume ambiguity
    :type sampled_ambiguity: 3D np.ndarray (row, col, eta)
    :param etas: range between eta_min and eta_max with step eta_step
    :type etas: np.ndarray
    :param nbr_etas: nuber of etas
    :type nbr_etas: int
    :param grids: array containing min and max disparity grids
    :type grids: 2D np.ndarray (min, max)
    :param disparity_range: array containing disparity range
    :type disparity_range: np.ndarray
    :param sample_risk: whether or not to compute and return the sampled risk
    :type sample_risk: bool
    :return: the risk and sampled risk if asked
    :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32, \
    3D np.ndarray (row, col) dtype = float32, 3D np.ndarray (row, col) dtype = float32)
    """
    return None, None
