#include <pybind11/pybind11.h>
#include "includes/refinement.hpp"
#include "includes/vfit.hpp"
#include "includes/quadratic.hpp"

namespace py = pybind11;

PYBIND11_MODULE(refinement_cpp, m) {
    m.doc() = "Refinement functions implemented in C++ with Pybind11";

    m.def(
        "loop_refinement", 
        &loop_refinement, 
        "Apply for each pixel the refinement method."
    );
    
    m.def(
        "loop_approximate_refinement", 
        &loop_approximate_refinement, 
        "Apply for each pixels the refinement method on the right disparity map which was created with the approximate method : a diagonal search for the minimum on the left cost volume"
    );

    m.def(
        "vfit_refinement_method", 
        &vfit_refinement_method,
        "Return the subpixel disparity and cost, by matching a symmetric V shape (linear interpolation)."
    );
    
    m.def(
        "quadratic_refinement_method", 
        &quadratic_refinement_method,
        "Return the subpixel disparity and cost, by matching a symmetric V shape (linear interpolation)."
    );

}