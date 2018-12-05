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
#include <time.h>

using namespace std;
using namespace cimg_library;

#define MAX(a,b)(((a)>(b))?(a):(b))

#define MIN(a,b)(((a)<(b))?(a):(b))

// Standard Functions
void edge_detection_wrapper(char flags, string input_name, string output_name, int threshold, int filter);

vector<int> vectorize_img(CImg<unsigned char> img);

vector<int> edge_detection_cpu(vector<int> img, int width, int height, int threshold, int filter);

int display_img(vector<int> img, int width, int height, int write_flag, string output);

int metadata(CImg<unsigned char> img, int threshold);

// CUDA Functions
void error_check(cudaError_t err);

__global__
void sobelFilterKernel(int *imageRGB, int *output, int width, int height, int Gx_matrix[][3], int Gy_matrix[][3], int threshold);
__global__
void robertFilterKernel(int *imageRGB, int *output, int width, int height, int RGx_array[][2], int RGy_array[][2], int threshold);
__global__
void prewittFilterKernel(int *imageRGB, int *output, int width, int height, int PGx_array[][3], int PGy_array[][3], int threshold);

vector<int> edge_detection_gpu(vector<int> img, int width, int height, int threshold, int filter);

// g-code Generator Functions

void gcode_prolog(void);

void gcode_epilog(void);

void next_to(int **image_2d, int **image_visited, int x, int y, int height, int width);

int surrounding_check(int x, int y, int last_x, int last_y);

bool next(int **image_2d, int **image_visited, int x, int y, int height, int width);

void g_gen(vector<int> img, int width, int height);
