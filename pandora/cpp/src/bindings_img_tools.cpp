#include <pybind11/pybind11.h>
#include "img_tools.hpp"

namespace py = pybind11;

PYBIND11_MODULE(img_tools_cpp, m) {
    m.doc() = "Image tools functions implemented in C++ with Pybind11";

    m.def(
        "interpolate_nodata_sgm", 
        &interpolate_nodata_sgm, 
        ""
    );

    m.def(
        "find_valid_neighbors", 
        &find_valid_neighbors, 
        ""
    );
    
}
