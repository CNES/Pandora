:mod:`pandora.refinement.refinement`
====================================

.. py:module:: pandora.refinement.refinement

.. autoapi-nested-parse::

   This module contains classes and functions associated to the subpixel refinement step.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pandora.refinement.refinement.AbstractRefinement



.. py:class:: AbstractRefinement

   Abstract Refinement class

   .. attribute:: __metaclass__
      

      

   .. attribute:: subpixel_methods_avail
      

      

   .. method:: subpixel_refinement(self, cv: xr.Dataset, disp: xr.Dataset) -> None

      Subpixel refinement of disparities and costs.

      :param cv: the cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv: xarray.Dataset
      :param disp: Dataset with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type disp: xarray.Dataset
      :return: None


   .. method:: approximate_subpixel_refinement(self, cv_left: xr.Dataset, disp_right: xr.Dataset) -> xr.Dataset

      Subpixel refinement of the right disparities map, which was created with the approximate method : a diagonal
      search for the minimum on the left cost volume

      :param cv_left: the left cost volume dataset with the data variables:

              - cost_volume 3D xarray.DataArray (row, col, disp)
              - confidence_measure 3D xarray.DataArray (row, col, indicator)
      :type cv_left: xarray.Dataset
      :param disp_right: right disparity map with the variables :

          - disparity_map 2D xarray.DataArray (row, col)
          - confidence_measure 3D xarray.DataArray (row, col, indicator)
          - validity_mask 2D xarray.DataArray (row, col)
      :type disp_right: xarray.Dataset
      :return: disp_right Dataset with the variables :

              - disparity_map 2D xarray.DataArray (row, col) that contains the refined disparities
              - confidence_measure 3D xarray.DataArray (row, col, indicator) (unchanged)
              - validity_mask 2D xarray.DataArray (row, col) with the value of bit 3 ( Information:                 calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
              - interpolated_coeff 2D xarray.DataArray (row, col) that contains the refined cost
      :rtype: xarray.Dataset


   .. method:: register_subclass(cls, short_name: str)
      :classmethod:

      Allows to register the subclass with its short name

      :param short_name: the subclass to be registered
      :type short_name: string


   .. method:: desc(self) -> None
      :abstractmethod:

      Describes the subpixel method
      :return: None


   .. method:: loop_refinement(cv: np.ndarray, disp: np.ndarray, mask: np.ndarray, d_min: int, d_max: int, subpixel: int, measure: str, method: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
      :staticmethod:

      Apply for each pixels the refinement method

      :param cv: cost volume to refine
      :type cv: 3D numpy array (row, col, disp)
      :param disp: disparity map
      :type disp: 2D numpy array (row, col)
      :param mask: validity mask
      :type mask: 2D numpy array (row, col)
      :param d_min: minimal disparity
      :type d_min: int
      :param d_max: maximal disparity
      :type d_max: int
      :param subpixel: subpixel precision used to create the cost volume
      :type subpixel: int ( 1 | 2 | 4 )
      :param measure: the measure used to create the cot volume
      :param measure: string
      :param method: the refinement method
      :param method: function
      :return: the refine coefficient, the refine disparity map, and the validity mask
      :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
       


   .. method:: refinement_method(self, cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]
      :abstractmethod:

      Return the subpixel disparity and cost

      :param cost: cost of the values disp - 1, disp, disp + 1
      :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
      :param disp: the disparity
      :type disp: float
      :param measure: the type of measure used to create the cost volume
      :type measure: string = min | max
      :return: the refined disparity (disp + sub_disp), the refined cost and the state of the pixel( Information:         calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
      :rtype: float, float, int


   .. method:: loop_approximate_refinement(cv: np.ndarray, disp: np.ndarray, mask: np.ndarray, d_min: int, d_max: int, subpixel: int, measure: str, method: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
      :staticmethod:

      Apply for each pixels the refinement method on the right disparity map which was created with the approximate
        method : a diagonal search for the minimum on the left cost volume

      :param cv: the left cost volume
      :type cv: 3D numpy array (row, col, disp)
      :param disp: right disparity map
      :type disp: 2D numpy array (row, col)
      :param mask: right validity mask
      :type mask: 2D numpy array (row, col)
      :param d_min: minimal disparity
      :type d_min: int
      :param d_max: maximal disparity
      :type d_max: int
      :param subpixel: subpixel precision used to create the cost volume
      :type subpixel: int ( 1 | 2 | 4 )
      :param measure: the type of measure used to create the cost volume
      :type measure: string = min | max
      :param method: the refinement method
      :type method: function
      :return: the refine coefficient, the refine disparity map, and the validity mask
      :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
       



