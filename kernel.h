//#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream> // C++ I/O library
#include "CImg.h"
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include<fstream>

using namespace std;
using namespace cimg_library;

void edge_detection_wrapper(char flags, string input_name, string output_name, int threshold);

vector<int> vectorize_img(CImg<unsigned char> img, int *width, int *height);

vector<int> edge_detection_cpu(vector<int> img, int width, int height, int threshold);

int display_img(vector<int> img, int width, int height, int write_flag, string output);

int metadata(CImg<unsigned char> img, int threshold);

void g_gen(int fd);
