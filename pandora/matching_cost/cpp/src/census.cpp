#include "census.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <bitset>
#include <bit>


namespace py = pybind11;

void censuswprint(char* chs, int init, int nb) {
    for (int i = 0; i < nb; i++) {
        std::cout << std::bitset< 8 >( chs[init + i] );
    }
    std::cout << std::endl;
}

char* census_transform(py::array_t<float> img, int c_w, int c_h) {

    auto r_img = img.unchecked<2>();
    int n_cols = r_img.shape(0);
    int n_rows = r_img.shape(1);

    int c_half_w = c_w / 2;
    int c_half_h = c_h / 2;

    int nb_useful_bits = c_w * c_h;
    int bits_per_elem = sizeof(char) * 8;
    int nb_chars = nb_useful_bits / bits_per_elem + nb_useful_bits % bits_per_elem;

    char* out = new char[n_rows*n_cols*nb_chars];
    for (int i = 0; i < n_rows*n_cols*nb_chars; ++i) out[i] = 0;

    int s = 0;
    
    for (int x = c_half_h; x < n_cols - c_half_h; ++x) {
        for (int y = c_half_w; y < n_rows - c_half_w; ++y) {

            // x,y -> all valid centers
            float val = r_img(x, y);
            int out_pos = x*nb_chars*n_rows + y*nb_chars;
            int bit = 0;
            int chr = 0;

            for (int wx = x-c_half_h; wx < x+c_half_h+1; ++wx) {
                for (int wy = y-c_half_w; wy < y+c_half_w+1; ++wy) {
                    
                    if (r_img(wx, wy) > val) {
                        out[out_pos + chr] += 0x80 >> bit;
                        s++;
                    } 

                    bit++;
                    if (bit >= bits_per_elem) {
                        chr++;
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

    int n_cols = (int) rw_cv.shape(0);
    int n_rows = (int) rw_cv.shape(1);
    int n_disp = (int) rw_cv.shape(2);

    int c_half_w = census_width / 2;
    int c_half_h = census_height / 2;

    int nb_useful_bits = census_width * census_height;
    int bits_per_elem = sizeof(char) * 8;
    int nb_chars = nb_useful_bits / bits_per_elem + nb_useful_bits % bits_per_elem;

    char* census_img_left = census_transform(img_left, census_width, census_height);

    std::vector<char*> census_imgs_right = std::vector<char*>(); 
    for (py::handle handle_img : imgs_right) {
        py::array_t<float> img = py::cast<py::array>(handle_img);
        census_imgs_right.push_back( census_transform(img, census_width, census_height) );
    }

    int id_right = 0;
    char* right_img = census_imgs_right[id_right];
    
    for (int col = c_half_h; col < n_cols-c_half_h; col++) {
        for (int row = c_half_w; row < n_rows-c_half_w; row++) {
            
            int left_pos = col*nb_chars*n_rows + row*nb_chars;
            for (int disp = 0; disp < n_disp; disp++) {
                
                int right_x = (row+disp+min_disp);
                if (right_x < c_half_w || right_x >= n_rows-c_half_w) continue;

                int right_pos = col*nb_chars*n_rows + right_x*nb_chars;
                int weight = 0;
                for (int chr = 0; chr < nb_chars; chr++) {
                    char xr = right_img[right_pos+chr] ^ census_img_left[left_pos+chr];
                    weight += std::bitset<8>(xr).count();
                }

                rw_cv(col, row, disp) = weight;
                
                id_right++;
                id_right %= census_imgs_right.size();
                right_img = census_imgs_right[id_right];

            }
        }
    }

    delete[] census_img_left;
    for (char* img : census_imgs_right) {
        delete[] img;
    }

    return cv;
}