#include "census.hpp"
#include <algorithm>
#include <climits> // For CHAR_BIT
#include <numeric>
#include <cmath>
#include <iostream>
#include <bitset>
#include <bit>

#define CENSUS_TYPE char
#define CENSUS_SINGLE_BIT (CENSUS_TYPE(1) << sizeof(CENSUS_TYPE)*CHAR_BIT - 1)

namespace py = pybind11;

CENSUS_TYPE* census_transform(py::array_t<float> img, int census_width, int census_height) {

    auto r_img = img.unchecked<2>();
    size_t n_rows = r_img.shape(0);
    size_t n_cols = r_img.shape(1);

    int c_half_w = census_width / 2;
    int c_half_h = census_height / 2;

    int nb_useful_bits = census_width * census_height;
    int bits_per_elem = sizeof(CENSUS_TYPE) * CHAR_BIT;
    int nb_chars = nb_useful_bits / bits_per_elem + nb_useful_bits % bits_per_elem;

    CENSUS_TYPE* out = new CENSUS_TYPE[n_cols*n_rows*nb_chars];
    for (int i = 0; i < n_cols*n_rows*nb_chars; ++i) out[i] = 0;

    for (int x = c_half_h; x < n_rows - c_half_h; ++x) {
        for (int y = c_half_w; y < n_cols - c_half_w; ++y) {

            // x,y -> all valid centers
            float val = r_img(x, y);
            int out_pos = x*nb_chars*n_cols + y*nb_chars;
            int bit = 0;
            int chr = 0;

            // for each image pixel (x,y), go through all the pixels in the window (wx, wy)
            for (int wx = x-c_half_h; wx < x+c_half_h+1; ++wx) {
                for (int wy = y-c_half_w; wy < y+c_half_w+1; ++wy) {
                    
                    // for the pixel (x,y), if the corresponding window pixel has a greater value,
                    // set the window pixel's bit to 1
                    if (r_img(wx, wy) > val) {
                        // CENSUS_SINGLE_BIT represents a CENSUS_TYPE with its highest bit set to 1
                        out[out_pos + chr] += CENSUS_SINGLE_BIT >> bit;
                    }

                    // bit always goes up
                    bit++;
                    if (bit >= bits_per_elem) {
                        // chr goes up once bit gets higher than the nb of bits in a CENSUS_TYPE
                        chr++;
                        // reset bit
                        bit = 0;
                    }

                }
            }
        }
    }

    return out;
}

py::array_t<float> compute_matching_costs(
    py::array_t<float> img_left,
    py::list imgs_right,
    py::array_t<float> cv,
    py::array_t<float> disps,
    size_t census_width,
    size_t census_height
) {

    auto rw_cv = cv.mutable_unchecked<3>();
    auto rw_disps = disps.mutable_unchecked<1>();

    int min_disp = lround(rw_disps(0));

    size_t n_rows = rw_cv.shape(0);
    size_t n_cols = rw_cv.shape(1);
    size_t n_disp = rw_cv.shape(2);

    int c_half_w = census_width / 2;
    int c_half_h = census_height / 2;

    int nb_useful_bits = census_width * census_height;
    int bits_per_elem = sizeof(CENSUS_TYPE) * CHAR_BIT;
    int nb_chars = nb_useful_bits / bits_per_elem + nb_useful_bits % bits_per_elem;

    CENSUS_TYPE* census_img_left = census_transform(img_left, census_width, census_height);

    std::vector<CENSUS_TYPE*> census_imgs_right = std::vector<CENSUS_TYPE*>(); 
    for (py::handle handle_img : imgs_right) {
        py::array_t<float> img = py::cast<py::array>(handle_img);
        census_imgs_right.push_back( census_transform(img, census_width, census_height) );
    }

    int subpix = census_imgs_right.size();
    CENSUS_TYPE* right_img;
    
    for (int row = c_half_h; row < n_rows-c_half_h; row++) {
        for (int col = c_half_w; col < n_cols-c_half_w; col++) {
            
            int left_pos = row*nb_chars*n_cols + col*nb_chars;
            for (int disp = 0; disp < n_disp; disp+=subpix) {
                
                int right_x = (col+disp/subpix+min_disp); // pixel
                if (right_x < c_half_w || right_x >= n_cols-c_half_w) continue;
                int right_pos = row*nb_chars*n_cols + right_x*nb_chars;

                for (int id_right = 0; (id_right < subpix) && (disp + id_right < n_disp); id_right++) {
                    
                    right_img = census_imgs_right[id_right];

                    int weight = 0;
                    for (int chr = 0; chr < nb_chars; chr++) {
                        CENSUS_TYPE xr = right_img[right_pos+chr] ^ census_img_left[left_pos+chr];
                        weight += std::bitset<sizeof(CENSUS_TYPE)*CHAR_BIT>(xr).count();
                    }

                    rw_cv(row, col, disp+id_right) = weight;
                }

            }
        }
    }

    delete[] census_img_left;
    for (CENSUS_TYPE* img : census_imgs_right) {
        delete[] img;
    }

    return cv;
}