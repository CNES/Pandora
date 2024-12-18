#include <pybind11/pybind11.h>
#include "includes/img_tools.hpp"

namespace py = pybind11;

PYBIND11_MODULE(img_tools_cpp, m) {
    m.doc() = "Image tools functions implemented in C++ with Pybind11";

    m.def(
        "interpolate_nodata_sgm", 
        &interpolate_nodata_sgm, 
        "Interpolation of the input image to resolve invalid (nodata) pixels.\n"
        "Interpolate invalid pixels by finding the nearest correct pixels in 8 different directions\n"
        "and use the median of their disparities.\n\n"
        "HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.\n"
        "IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341."
    );

    m.def(
        "find_valid_neighbors", 
        &find_valid_neighbors, 
        "Find valid neighbors along directions"
    );
    
}
