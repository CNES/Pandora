#include <pybind11/pybind11.h>
#include "includes/interpolated_disparity.hpp"

namespace py = pybind11;

PYBIND11_MODULE(validation_cpp, m) {
    m.doc() = "Validation functions implemented in C++ with Pybind11";
    
    m.def(
        "interpolate_occlusion_sgm", 
        &interpolate_occlusion_sgm, 
        "Calculate the horizontal integral image."
    );
    m.def(
        "interpolate_mismatch_sgm", 
        &interpolate_mismatch_sgm, 
        "Calculate the horizontal integral image."
    );
    
    m.def(
        "interpolate_occlusion_mc_cnn", 
        &interpolate_occlusion_mc_cnn, 
        "Calculate the horizontal integral image."
    );
    m.def(
        "interpolate_mismatch_mc_cnn", 
        &interpolate_mismatch_mc_cnn, 
        "Calculate the horizontal integral image."
    );

}