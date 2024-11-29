#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j + 2;
}

PYBIND11_MODULE(pandora_lib_c, m) {
    m.doc() = "Pandora's pybind11 module"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}