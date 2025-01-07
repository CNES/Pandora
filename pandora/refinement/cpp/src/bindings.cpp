#include <pybind11/pybind11.h>
#include "refinement.hpp"
#include "vfit.hpp"
#include "quadratic.hpp"

namespace py = pybind11;

PYBIND11_MODULE(refinement_cpp, m) {
    m.doc() = "Refinement functions implemented in C++ with Pybind11";

    m.def(
        "loop_refinement", 
        &loop_refinement, 
        ""
    );
    
    m.def(
        "loop_approximate_refinement", 
        &loop_approximate_refinement, 
        ""
    );

    m.def(
        "vfit_refinement_method", 
        &vfit_refinement_method,
        ""
    );
    
    m.def(
        "quadratic_refinement_method", 
        &quadratic_refinement_method,
        ""
    );

}