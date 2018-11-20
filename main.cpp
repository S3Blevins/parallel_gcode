#include "kernel.h"         // Contains function prototypes for CUDA kernels
#include <iostream>         // C++ I/O library
#include "CImg.h"  // For IMG handling

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
    CImg<unsigned int> img;

    try {
        img.assign(argv[1]);
    } catch (CImgIOException) {
        cout << "Image file has not been located. Please use an appropriate image." << endl;
        exit(0);
    }

    int width = img.width();
    int height = img.height();

    int c;              // switch case variable
    while((c = getopt(argc, argv, "w:th")) != -1) {
        switch(c) {
            case 'w':   // creates a file to write to
                    file_name = optarg;

                    int sum;
                    // shorthand forloop completed by library
                    cimg_forXY(img, x, y) {
                        sum = 0;
                        sum += img.atXY(x,y,0); // red channel
                        sum += img.atXY(x,y,1); // blue channel
                        sum += img.atXY(x,y,2); // green channel
                        printf("%d ", sum);
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


    return 0;
}
