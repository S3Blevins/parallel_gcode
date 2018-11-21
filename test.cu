#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

void error_check(cudaError_t err);
__global__ void sobelFilterKernel(int width, int height, int *imageRGB, int *Gx, int *Gy);

int main()
{
    int width;
    int height;
    int size_picture;

    //Get the paramters of the picture here. Hard coded for now.
    width = 512;
    heaight = 512;
    size_picture = width * height;

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



/*
Thread block size should always be a multiple of 32. This is to maximize
efficientcy of utilizing all threads. Example:

If Block size = 50, then we need (32 threads per block) = (2 * 32 to cover 50) = 64
                    This means we are wasting 14 threads.
*/

// SobelFilter <<< # blocks in grid , # of threads in blocks >>> ()
__global__ void sobelFilterKernel(int width, int height, int *imageRGB, int *Gx, int *Gy)
{


}

/**
 * function will error check the cuda malloc.
 * If error, system will output where.
 * @params: err which is if the cuda malloc worked or not.
 */
void error_check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
         exit(EXIT_FAILURE);
    }
}
