#include "kernel.h"         // Contains function prototypes for CUDA kernels
#include <iostream>         // C++ I/O library
#include "CImg.h"  // For IMG handling
#include <cmath>

using namespace cimg_library;
using namespace std;

void helper(void) {
    printf("The main functionality of the program...\n");
}

int main(int argc, char *argv[]) {
    cimg::exception_mode(0);    // silence library exceptions so we can use our own

    int fd;             // file descriptor for file to analyze
    int cd;             // file descriptor for file to create
    char *file_name;

    // open a file
    CImg<unsigned char> img;

    try {
        img.assign(argv[1]);
    } catch (CImgIOException) {
        cout << "Image file has not been located. Please use an appropriate image." << endl;
        exit(0);
    }

    int width = img.width();
    int height = img.height();
/*
    int** matrix;
    matrix = new int*[width];
    for(int i = 0; i < width; i++) {
        matrix[i] = new int[height];
    }
*/

    int Gx;
    int Gy;
    int p;
    float length = 0;
CImg<unsigned char> img_new;
img_new.assign(width, height, 1, 3);


for(int x = 1; x < width; x++) {
    for(int y = 1; y < height; y++) {

        // initialize Gx to 0 and Gy to 0 for every pixel
        Gx = 0;
        Gy = 0;

        // top left
        p = img.atXY(x-1,y-1,0); // red channel
        p += img.atXY(x-1,y-1,1); // blue channel
        p += img.atXY(x-1,y-1,2); // green channel

        // intensity ranges from 0 to 765 (255 * 3)
        // accumulate the value into Gx, and Gy
        Gx += -p;
        Gy += -p;

        // left middle
        p = img.atXY(x-1,y,0); // red channel
        p += img.atXY(x-1,y,1); // blue channel
        p += img.atXY(x-1,y,2); // green channel

        Gx += -2 * p;

        // left bottom
        p = img.atXY(x-1,y+1,0); // red channel
        p += img.atXY(x-1,y+1,1); // blue channel
        p += img.atXY(x-1,y+1,2); // green channel

        Gx += -p;
        Gy += p;

        // middle Top
        p = img.atXY(x,y-1,0); // red channel
        p += img.atXY(x,y-1,1); // blue channel
        p += img.atXY(x,y-1,2); // green channel

        Gy += -2 * p;

        // middle bottom
        p = img.atXY(x,y+1,0); // red channel
        p += img.atXY(x,y+1,1); // blue channel
        p += img.atXY(x,y+1,2); // green channel

        Gy += 2 * p;

        // top right
        p = img.atXY(x+1,y-1,0); // red channel
        p += img.atXY(x+1,y-1,1); // blue channel
        p += img.atXY(x+1,y-1,2); // green channel

        Gx = p;
        Gy += -p;

        // middle right
        p = img.atXY(x+1,y,0); // red channel
        p += img.atXY(x+1,y,1); // blue channel
        p += img.atXY(x+1,y,2); // green channel

        Gx += 2 * p;

        // bottom right
        p = img.atXY(x+1,y+1,0); // red channel
        p += img.atXY(x+1,y+1,1); // blue channel
        p += img.atXY(x+1,y+1,2); // green channel

        Gx += p;
        Gy += p;

        // calculate the length of the gradient (Pythagorean theorem)
        length = sqrt((Gx*Gx) + (Gy*Gy));

        // normalise the length of gradient to the range 0 to 255


        float test = length / 4328 * 255;
        int new_test = (int) test;
        printf("test = %d\n", new_test);
        //draw the length in the edge image
        img_new(x,y,0,0)=new_test;
        img_new(x,y,0,1)=new_test;
        img_new(x,y,0,2)=new_test;
    }
}




    //vector<vector<int>> matrix = new int[width][height];

    int c;              // switch case variable
    while((c = getopt(argc, argv, "w:th")) != -1) {
        switch(c) {
            case 'w':   // creates a file to write to
                    file_name = optarg;

                    int sum;
                    // shorthand forloop completed by library
                    /*
                    cimg_forXY(img, x, y) {
                        sum = 0;
                        sum += img.atXY(x,y,0); // red channel
                        sum += img.atXY(x,y,1); // blue channel
                        sum += img.atXY(x,y,2); // green channel

                        printf("%d ", sum);
                    }
                    */
                    for(int x = 0; x < height; x++) {
                        for(int y = 0; y < width; y++) {
                            sum = 0;
                            sum += img.atXY(x,y,0); // red channel
                            sum += img.atXY(x,y,1); // blue channel
                            sum += img.atXY(x,y,2); // green channel
                            //matrix[x][y] = sum;
                        }
                    }

                    //cout << img.data() << endl;
                    //vector<int> matrix = img.vector();
                    break;
            case 't':
                    cout << img.spectrum() << endl;
                    printf("Width: %i\n", width);
                    printf("Height: %i\n", height);
                    printf("Total Pixel Count: %.2f MP\n", ((float)(width * height))/1000000);
                    break;
            case 'h':
                    //helper();
                    break;
            default:                // defaults to usage again
                    //helper();
                    break;
            }
    }

    CImgDisplay main_disp(img_new,"Image");
    while (!main_disp.is_closed())
        main_disp.wait();

    return 0;
}
