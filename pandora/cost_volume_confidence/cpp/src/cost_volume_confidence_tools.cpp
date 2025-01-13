#include "cost_volume_confidence_tools.hpp"

size_t searchsorted(const py::array_t<float>& array, float value) {
    auto arr = array.unchecked<1>();

    size_t left = 0;
    size_t right = arr.shape(0) - 1;
    size_t mid;

    while (left < right) {
        mid = left + (right - left) / 2;

        if (arr(mid) < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}
