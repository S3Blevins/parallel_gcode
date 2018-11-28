#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream> // C++ I/O library
#include "CImg.h"
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include<fstream>
#include<string>

using namespace std;
using namespace cimg_library;

// Standard Functions
void edge_detection_wrapper(char flags, string input_name, string output_name, int threshold);

vector<int> vectorize_img(CImg<unsigned char> img, int *width, int *height);

vector<int> edge_detection_cpu(vector<int> img, int width, int height, int threshold);

int display_img(vector<int> img, int width, int height, int write_flag, string output);

int metadata(CImg<unsigned char> img, int threshold);

// CUDA Functions
void error_check(cudaError_t err);

__global__
void sobelFilterKernel(int *imageRGB, int *output, int width, int height, int Gx_matrix[][3], int Gy_matrix[][3], int threshold);

vector<int> edge_detection_gpu(vector<int> img, int width, int height, int threshold);

// g-code Generator Functions
void g_gen(vector<int> img, int width, int height);
