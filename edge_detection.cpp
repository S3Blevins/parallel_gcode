//#include <cuda.h>
#include "kernel.h"
#include <queue>
#include <fstream>

using namespace cimg_library;
using namespace std;

#define BUF 4096
/*
    NOTE: The cuda kernels do not have issues with 2d arrays that are declared
    before compile time
*/

// sobelFilter x matrix
// (PREDEFINED AT COMPILE TIME)
int Gx_matrix[3][3] = {
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
};

// sobelFilter y matrix
// (PREDEFINED AT COMPILE TIME)
int Gy_matrix[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

// GLOBAL
ofstream outputFile("gcode_out_test.gcode");

int gcode(vector<int> image, int width, int height);

/**
 * edge detection wrapper processes the appropriate flags
 * @param flags       flags to execute
 * @param input_name  image name
 * @param output_name output name for gcode output
 * @param threshold   threshold used for normalizing the sobel filter
 */
void edge_detection_wrapper(char flags, string input_name, string output_name, int threshold) {
    cimg::exception_mode(0); // silence library exceptions so we can use our own

    CImg<unsigned int> img;

    // dimensions to be set by set when cast by reference below
    int width;
    int height;

    // open image
    try {
        img.assign(input_name.c_str());
    } catch (CImgIOException) {
        cout << "Image file has not been located. Please use an appropriate image." << endl;
        exit(0);
    }

    // vectorize the image
    vector<int> image_vector = vectorize_img(img, &width, &height);

    // print out some minor metadata (need to finish)
    if(flags & 0x8) {
        metadata(img, threshold);
    }

    // if GPU or CPU processed, call the appropriate function
    if(flags & 0x1) {
        //printf("GPU Processed\n");
    } else {
        //printf("CPU Processed\n");

        // overwrite the image vector with sobel filter
        //image_vector = edge_detection_cpu(image_vector, width, height, threshold);
    }

    gcode(image_vector, width, height);

    // display the image when the filter has been applied
    if(flags & 0x2) {
        display_img(image_vector, width, height, (flags & 0x10), output_name);
    }
}

/*
void error_check(cudaError_t err);

__global__
void sobelFilterKernel(int width, int height, int *imageRGB, int *Gx, int *Gy);

int edge_detection_gpu(int width, int height, int threshold) {

    int size_picture = width * height;

    int picture_memory = size_picture * sizeof(int);

    //Looking the each matrix as a linear 2-D array.
    //Populate the matricies (simplified, change to accomidate other filters later)
    int sobelFilter_x[9] = {-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1};
   int sobelFilter_y[9] = {-1, -2, -1,
                            0, 0, 0,
                            1, 2, 1};

    //Start allocating memory for device variables
    //IMG_array is the rgb values of the picture.
    int *IMG_array = (int *) malloc(picture_memory);
    if (IMG_array == NULL) {
        printf("Could not allocate memory for IMG_array: failed\n");
        exit(1);
    }
    int *sobelPictureOutput = (int *) malloc(picture_memory);
    if (sobelPictureOutput == NULL) {
        printf("Could not allocate memory for sobelPictureOutput: failed\n");
        exit(1);
    }

    //Populate the IMG_array here

    //Device IMG_array, device Sobel Filter x, device Sobel Filter y
    int *outputIMG_array, dSFx, dSFy;

    //allocating memory for device variables
    //--------------------------------------------------------------------------
    cudaError_t err = cudaMalloc((void **) &outputIMG_array, picture_memory);
    error_check(err);

    err = cudaMalloc((void **) &dSFx, picture_memory);
    error_check(err);

    err = cudaMalloc((void **) &dSFy, picture_memory);
    error_check(err);
    //--------------------------------------------------------------------------

    //Copy dSFx and dSFy to device memory.
    cudaMemcpy(dSFx, sobelFilter_x, picture_memory, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFy, sobelFilter_y, picture_memory, cudaMemcpyHostToDevice);

    //Launch kernel
    sobelFilterKernel <<< 1, 1, >>> (width, height, outputIMG_array, dSFx, dSFy);


    //Success - This point should have the picture in an output array
    cudaMemcpy(sobelPictureOutput, outputIMG_array, picture_memory, cudaMemcpyDeviceToHost);


    //Here we have sobelPictureOutput that we need to print and show the results

    return 0;
}
*/
/*
Thread block size should always be a multiple of 32. This is to maximize
efficientcy of utilizing all threads. Example:

If Block size = 50, then we need (32 threads per block) = (2 * 32 to cover 50) = 64
                    This means we are wasting 14 threads.
*/
/*
// SobelFilter <<< # blocks in grid , # of threads in blocks >>> ()
__global__
void sobelFilterKernel(int width, int height, int *imageRGB, int *Gx, int *Gy) {

}
*/
/**
 * function will error check the cuda malloc.
 * If error, system will output the location.
 * @params: err which is if the cuda malloc worked or not.
 */
/*
void error_check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}
*/

/**
 * image is turned into a vector (un-avoidable overhead)
 * @param  img    image to be converted into a 1D vector
 * @param  width  width is created via pass by reference
 * @param  height height is created via pass by reference
 * @return        vectorized image in terms of integers
 */
vector<int> vectorize_img(CImg<unsigned char> img, int *width, int *height) {
    *width = img.width();
    *height = img.height();

    vector<int> image_vector((*width) * (*height));

    // loop through pixels x and y
    for(int x = 0; x < *width; x++) {
        for(int y = 0; y < *height; y++) {
           image_vector[x + (y * (*width))] = img.atXY(x,y);
           //printf("%i\n", image_vector[x + (y * (*width))]);
        }
    }

    return image_vector;
}

/**
 * Sobel Edge Detection filter run via the CPU.
 * @param  width     width of the image
 * @param  height    height of the image
 * @param  threshold threshold for normalization for filter
 * @return           vectorized image with filter applied
 */
vector<int> edge_detection_cpu(vector<int> img, int width, int height, int threshold) {

    int Gx;
    int Gy;

    int length;
    int normalized_pixel;

    vector<int> image_vector(width * height);

    // loop through pixels x and y
    for(int x = 1; x < width; x++) {
        for(int y = 1; y < height; y++) {
            // initialize Gx and Gy intensities to 0 for every pixel
            Gx = 0;
            Gy = 0;
            int RGB;;

            // loop through the filter matrices
            for(int col = 0; col < 3; col++) {
                for(int row = 0; row < 3; row++) {

                    // make index correction for pixels surrounding x and y
                    // img.atXY(x + i - 1 , y + j - 1)
                    RGB = img[(x + col - 1) + (width * (y + row - 1))];

                    // summation of Gx and Gy intensities
                    Gx += Gx_matrix[col][row] * RGB;
                    Gy += Gy_matrix[col][row] * RGB;
                }
            }
            // absolute value of intensities
            length = abs(Gx) + abs(Gy);

            // normalize the gradient with threshold value (DEFAULT: 2048)
            normalized_pixel = length * 255 / threshold;

            // set pixel value
            image_vector[x + (width * y)] = normalized_pixel;
        }
    }

    return image_vector;
}

/**
 * Used when the flag to view the image is used
 * @param  img        vectorized image
 * @param  width      width of image
 * @param  height     height of image
 * @param  write_flag flag for writing new image to a file
 * @param  output     output name
 * @return            0
 */
int display_img(vector<int> img, int width, int height, int write_flag, string output) {

    CImg<unsigned char> new_img;
    new_img.assign(width, height, 1, 3);

    for(int x = 0; x < width; x++) {
        for(int y = 0; y < height; y++) {
            // Red, Green, and Blue values are all the same
            new_img.atXY(x,y,0) = img[x + (y * width)];
            new_img.atXY(x,y,1) = img[x + (y * width)];
            new_img.atXY(x,y,2) = img[x + (y * width)];
        }
    }

    // Display the image
    CImgDisplay main_disp(new_img,"Image");
    while (!main_disp.is_closed()) {
        main_disp.wait();
    }

    // if write_flag exists, then save the image
    if(write_flag) {
        output.append(".bmp");
        new_img.save(output.c_str());
    }

    return 0;
}
/*

// write the x and y coordinates from a position on a
// 1d array where x and y are passed by reference because I didn't
// know whether or not to use tuples
void converter_xy(int position, int *x, int *y, int width) {

    *x = position % width;
    *y = position - *x / width;
}

// converts an x and y coordinate into a
// position on a 1d array
int converter_1d(int x, int y, int width) {

    return x + (y * (width));
}
*/
void gcode_primer(void) {
    // G1 means to extrude
    // G0 means to not extrude
    // comments are denoted by a semicolon

    outputFile << "M190 S50.000000" << endl;
    outputFile << "M109 S215.000000" << endl << endl;

    outputFile << "G21            ;metric values" << endl;
    outputFile << "G90            ;absolute positioning" << endl;
    outputFile << "M82            ;set extruder to absolute mode" << endl;
    outputFile << "M107           ;start with the fan off" << endl;
    outputFile << "G28 X0 Y0      ;move X/Y to min endstops" << endl;
    outputFile << "G28 Z0         ;move Z to min endstops" << endl;
    outputFile << "G0 Z15.0 F9000 ;move the platform down 15mm" << endl;
    outputFile << "G92 E0         ;zero the extruded length" << endl;
    outputFile << "G1 F9000       ;Put printing message on LCD screen" << endl;
    outputFile << "M117 DRAWING..." << endl << endl;

    outputFile << ";Layer count: 1" << endl;
    outputFile << ";LAYER:0" << endl;
    outputFile << "M107           ;Turn off the fan" << endl;

    // actual gcode goes below;
    // G0 {speed} X{position} Y{position}


}
void next_to(int **image_2d, int **image_visited, int x, int y) {

    int new_x;
    int new_y;

    //image_visited[x][y] = 1;
    //printf("original pixel\n");
    //printf("pixel[%d][%d] = %d\n", x, y, image_2d[x][y]);
    // look at all pixels surrounding the main pixel in question
    //printf("checking pixels...\n");
    for(int col = 0; col < 3; col++) {
        for(int row = 0; row < 3; row++) {
            new_x = x + col - 1;
            new_y = y + row - 1;
            //printf("checking pixel[%d][%d] = %d\n", new_x, new_y, image_visited[new_x][new_y]);
            if(image_2d[new_x][new_y] >= 50 && image_visited[new_x][new_y] == 0) {
                image_visited[new_x][new_y] = 1;
                printf("pixel[%d][%d] = %d\n", new_x, new_y, image_2d[new_x][new_y]);
                outputFile << "G0 X" << new_x << "Y" << new_y << endl;
                next_to(image_2d, image_visited, new_x, new_y);
            }
        }
    }
}

int gcode(vector<int> image, int width, int height) {

    int **image_2d;
    image_2d = new int *[width];
    int **image_visited;
    image_visited = new int *[width];

    gcode_primer();

    // rebuild the image in 2d format
    for(int i = 0; i < width; i++) {
        image_2d[i] = new int[height];
        image_visited[i] = new int[height];
        for(int j = 0; j < height; j++) {
            image_2d[i][j] = image[i + (j * (width))];
            image_visited[i][j] = 0;
        }
    }

    // image_2d will have a normal color array where anything above
    // a 50 is considered a path.

    // image_visited will have a 0 if the item has not been visited
    // and a 1 if the image has been visited.

     // iterate through the 2d array(s)
     for(int x = 1; x < (width - 1); x++) {
         for(int y = 1; y < (height - 1); y++) {
             // if the image is grey/white and has not been visited
             if(image_2d[x][y] >= 50 && image_visited[x][y] == 0) {
                //printf("pixel[%d][%d] = %d\n", x, y, image_2d[x][y]);
                 // recursive call once a grey/white pixel has been found
                 // and follow up with any pixels which are grey/white
                 // immediately next to that
                next_to(image_2d, image_visited, x, y);
             }
         }
     }

    return 0;
}

/**
 * metadata of image
 * @param  img image object
 * @return     0
 */
int metadata(CImg<unsigned char> img, int threshold) {

    // TODO: add more useful data

    // populate dimensions
    int width = img.width();
    int height = img.height();

    // print out dimensions, etc
    printf("Width: %i\n", width);
    printf("Height: %i\n", height);
    printf("Total Pixel Count: %.2f MP\n", ((float)(width * height))/1000000);
    printf("Threshold value: %d\n", threshold);

    return 0;

}
