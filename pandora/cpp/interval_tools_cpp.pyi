# pylint: skip-file
import numpy as np
from typing import Tuple

def create_connected_graph(border_left: np.ndarray, border_right: np.ndarray, depth: int) -> np.ndarray: ...
def graph_regularization(
    interval_inf: np.ndarray,
    interval_sup: np.ndarray,
    border_left: np.ndarray,
    border_right: np.ndarray,
    connection_graph: np.ndarray,
    quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
