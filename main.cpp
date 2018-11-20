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

    int intensity;

for(int x = 1; x < width; x++) {
    for(int y = 1; y < height; y++) {

        // initialize Gx to 0 and Gy to 0 for every pixel
        Gx = 0;
        Gy = 0;

        // top left pixel
        p = img.atXY(x-1,y-1,0); // red channel
        p += img.atXY(x-1,y-1,1); // blue channel
        p += img.atXY(x-1,y-1,2); // green channel

        // intensity ranges from 0 to 765 (255 * 3)
        // accumulate the value into Gx, and Gy
        Gx += -p;
        Gy += -p;

        // remaining left column
        p = img.atXY(x-1,y,0); // red channel
        p += img.atXY(x-1,y,1); // blue channel
        p += img.atXY(x-1,y,2); // green channel

        Gx += -2 * p;

        p = img.atXY(x-1,y+1,0); // red channel
        p += img.atXY(x-1,y+1,1); // blue channel
        p += img.atXY(x-1,y+1,2); // green channel

        Gx += -p;
        Gy += p;

        // middle pixels
        p = img.atXY(x,y-1,0); // red channel
        p += img.atXY(x,y-1,1); // blue channel
        p += img.atXY(x,y-1,2); // green channel

        Gy += -2 * p;

        p = img.atXY(x,y+1,0); // red channel
        p += img.atXY(x,y+1,1); // blue channel
        p += img.atXY(x,y+1,2); // green channel

        Gy += 2 * p;

        # right column
        p = img.getpixel((x+1, y-1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += (r + g + b)
        Gy += -(r + g + b)

        p = img.getpixel((x+1, y))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += 2 * (r + g + b)

        p = img.getpixel((x+1, y+1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += (r + g + b)
        Gy += (r + g + b)

        # calculate the length of the gradient (Pythagorean theorem)
        length = math.sqrt((Gx * Gx) + (Gy * Gy))

        # normalise the length of gradient to the range 0 to 255
        length = length / 4328 * 255

        length = int(length)

        # draw the length in the edge image
        #newpixel = img.putpixel((length,length,length))
        newimg.putpixel((x,y),(length,length,length))
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
                            matrix[x][y] = sum;
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


    return 0;
}
