:mod:`pandora.refinement.vfit`
==============================

.. py:module:: pandora.refinement.vfit

.. autoapi-nested-parse::

   This module contains functions associated to the vfit method used in the refinement step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.refinement.vfit.Vfit



.. py:class:: Vfit(**cfg: str)

   Bases: :class:`pandora.refinement.refinement.AbstractRefinement`

   Vfit class allows to perform the subpixel cost refinement step

   .. method:: check_conf(**cfg: str) -> Dict[str, str]
      :staticmethod:

      Add default values to the dictionary if there are missing elements and check if the dictionary is correct

      :param cfg: refinement configuration
      :type cfg: dict
      :return cfg: refinement configuration updated
      :rtype: dict


   .. method:: desc(self) -> None

      Describes the subpixel refinement method
      :return: None


   .. method:: refinement_method(cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]
      :staticmethod:

      Return the subpixel disparity and cost, by matching a symmetric V shape (linear interpolation)

      :param cost: cost of the values disp - 1, disp, disp + 1
      :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
      :param disp: the disparity
      :type disp: float
      :param measure: the type of measure used to create the cost volume
      :param measure: string = min | max
      :return: the refined disparity (disp + sub_disp), the refined cost and the state of the pixel( Information:         calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
      :rtype: float, float, int



