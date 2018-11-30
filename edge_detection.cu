//#include <cuda.h>
#include "kernel.h"
#include "gcode_gen.cu"
#include <stdio.h>

using namespace cimg_library;
using namespace std;

// sobelFilter x matrix
// (PREDEFINED AT COMPILE TIME)
int Gx_matrix[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

// sobelFilter y matrix
// (PREDEFINED AT COMPILE TIME)
int Gy_matrix[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

/*
Thread block size should always be a multiple of 32. This is to maximize
efficientcy of utilizing all threads. Example:

If Block size = 50, then we need (32 threads per block) = (2 * 32 to cover 50) = 64
                    This means we are wasting 14 threads.
*/
__global__
void sobelFilterKernel(int *imageRGB, int *output, int width, int height, int *Gx_array, int *Gy_array, int threshold) {
    int Gx, Gy;
    int length;
    int normalized_pixel;

    //calculate thread locations (threadIDx)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int x = i % width;
    int y = i / width;

    // loop through pixels x and y
    //for(int x = 1; x < width; x++) {
    //    for(int y = 1; y < height; y++) {

    // initialize Gx and Gy intensities to 0 for every pixel
    Gx = 0;
    Gy = 0;
    int RGB;

    if((i < (width * height)) && (x > 0) && (y > 0)  && (x < width - 1) && (y < height - 1)) {

        for(int filter_pos = 0; filter_pos < 9; filter_pos++) {
            int col = filter_pos % 3;
            int row = filter_pos / 3;

            RGB = imageRGB[(x + col - 1) + (width * (y + row - 1))];

            // summation of Gx and Gy intensities
            Gx += Gx_array[filter_pos] * RGB;
            Gy += Gy_array[filter_pos] * RGB;
        }

        // absolute value
        if(Gx < 0) {
            Gx *= -1;
        }
        if(Gy < 0) {
            Gy *= -1;
        }

        // absolute value of intensities
        length = Gx + Gy;

        // normalize the gradient with threshold value (DEFAULT: 2048)
        normalized_pixel = length * 255 / threshold;
        __syncthreads();
        output[x + (width * y)] = normalized_pixel;
    }
     __syncthreads();
}

/**
 * function will error check the cuda malloc.
 * If error, system will output the location.
 * @params: err which is if the cuda malloc worked or not.
 */
void error_check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

/**
 * edge detection wrapper processes the appropriate flags
 * @param flags       flags to execute
 * @param input_name  image name
 * @param output_name output name for gcode output
 * @param threshold   threshold used for normalizing the sobel filter
 */
void edge_detection_wrapper(char flags, string input_name, string output_name, int threshold, int filter) {
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

        // overwrite the image vector with sobel filter
        image_vector = edge_detection_gpu(image_vector, width, height, threshold, filter);
    } else {
        //printf("CPU Processed\n");

        // overwrite the image vector with sobel filter
        image_vector = edge_detection_cpu(image_vector, width, height, threshold, filter);
    }

    // call g-code generator here
    // needs to be linked in makefile
     //g_gen(image_vector, width, height, output_name);

    // display the image when the filter has been applied
    if(flags & 0x2 || flags & 0x10) {
        display_img(image_vector, width, height, (flags), output_name);
    }

}

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

vector<int> edge_detection_gpu(vector<int> img, int width, int height, int threshold, int filter) {

    int image_size = width * height;
    int image_array_size = image_size * sizeof(int);

    int matrix_array_size = 9 * sizeof(int);

    // NOTE: we can convert a vector into an integer array needed
    // for the kernel by int* img_array = &img[0]
    // which makes a pointer from the first index which is then
    // continuously allocated.

    // convert vector into standard image array
    int* img_array = &img[0];

    // Device IMG_array, device Sobel Filter x, device Sobel Filter y
    int *inputIMG_array, *outputIMG_array, *sobelx, *sobely;

    // allocating memory for device variables
    //--------------------------------------------------------------------------
    cudaError_t err = cudaMalloc((void **) &outputIMG_array, image_array_size);
    error_check(err);

    err = cudaMalloc((void **) &inputIMG_array, image_array_size);
    error_check(err);

    err = cudaMalloc((void **) &sobelx, matrix_array_size);
    error_check(err);

    err = cudaMalloc((void **) &sobely, matrix_array_size);
    error_check(err);
    //--------------------------------------------------------------------------

    // Copy array to device memory
    cudaMemcpy(inputIMG_array, img_array, image_array_size, cudaMemcpyHostToDevice);

    // Copy array to device memory
    cudaMemcpy(sobelx, Gx_matrix, matrix_array_size, cudaMemcpyHostToDevice);

    // Copy array to device memory
    cudaMemcpy(sobely, Gy_matrix, matrix_array_size, cudaMemcpyHostToDevice);

    // Launch kernel (UNSURE OF BLOCKS PER GRID vs THREADS PER BLOCK)
    sobelFilterKernel <<< ceil(image_size/256.0), 256 >>> (inputIMG_array, outputIMG_array, width, height, sobelx, sobely, threshold);

cudaDeviceSynchronize();

    // Start allocating memory for new device variables
    int *sobelImageOutput;
//262144

    // sobelPictureOutput is the RGB values of the image normalized to sobelFilter
    sobelImageOutput = (int *) malloc(image_array_size);
    if (sobelImageOutput == NULL) {
        printf("Could not allocate memory for sobelPictureOutput: failed\n");
        exit(1);
    }


    // Success - This point should have the picture in an output array
    cudaMemcpy(sobelImageOutput, outputIMG_array, image_array_size, cudaMemcpyDeviceToHost);


    // Here we have sobelPictureOutput that we need to print and show the results

    // I think this is how we convert an array into a vector?
    vector<int> out(sobelImageOutput, sobelImageOutput + image_size);
    return out;
}

/**
 * Sobel Edge Detection filter run via the CPU.
 * @param  width     width of the image
 * @param  height    height of the image
 * @param  threshold threshold for normalization for filter
 * @return           vectorized image with filter applied
 */
vector<int> edge_detection_cpu(vector<int> img, int width, int height, int threshold, int filter) {

    int Gx;
    int Gy;

    int col;
    int row;

    int RGB;
    int length;
    int normalized_pixel;

    vector<int> image_vector(width * height);

    // loop through pixels x and y
    for(int x = 1; x < width; x++) {
        for(int y = 1; y < height; y++) {
            // initialize Gx and Gy intensities to 0 for every pixel
            Gx = 0;
            Gy = 0;

            for(int i = 0; i < 9; i++) {
                col = i % 3;
                row = i / 3;

                RGB = img[(x + col - 1) + (width * (y + row - 1))];

                // summation of Gx and Gy intensities
                Gx += Gx_matrix[i] * RGB;
                Gy += Gy_matrix[i] * RGB;
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
int display_img(vector<int> img, int width, int height, int flags, string output) {

    CImg<unsigned char> new_img;
    new_img.assign(width, height, 1, 3);

    for(int x = 0; x <= width; x++) {
        for(int y = 0; y <= height; y++) {
            // Red, Green, and Blue values are all the same
            new_img.atXY(x,y,0) = img[x + (y * width)];
            new_img.atXY(x,y,1) = img[x + (y * width)];
            new_img.atXY(x,y,2) = img[x + (y * width)];
            if(new_img.atXY(x,y) >= 50) {
                //printf("new_img.atXY[%d][%d] = %d\n", x, y, new_img.atXY(x,y));
            }
        }
    }

    // Display the image
    if (flags & 0x2) {
        CImgDisplay main_disp(new_img,"Image");
        while (!main_disp.is_closed()) {
            main_disp.wait();
        }
    }

    // if write_flag exists, then save the image
    if(flags & 0x10) {
        output.append(".bmp");
        new_img.save(output.c_str());
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
