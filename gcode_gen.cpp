// Gcode Gen class
#include <iostream>
#include <math.h>
#include <string>
#include "kernel.h"
#include "stdio.h"

using namespace std;

struct pixel{
    int *value;
    int* next_p;
    int* prev_p;
    int* down_p;
    int* left_down;
    int* right_down;
};

void g_gen(int fd)
{
    int matrix[5][5] = {{0, 1, 0, 1, 1},
                        {0, 1, 1, 0, 0},
                        {1, 1, 0, 0, 0},
                        {1, 0, 0, 0, 0},
                        {1, 1, 0, 0, 0}};

    struct pixel pix;
    int rows = sizeof(matrix[0])/sizeof(matrix[0][0]);
    int cols = sizeof(matrix)/sizeof(matrix[0]);

    pix.value = &matrix[0][0];
    pix.next_p = pix.value + 1;

    // traversing the matrix
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
        {
            if(!*pix.value)
            {
                pix.prev_p = pix.value;
                pix.value = pix.next_p;
                pix.next_p++;
                continue;
            }

            printf("opened, (%d, %d): %d\n", row,col, *pix.value);
            pix.prev_p = pix.value;
            pix.value = pix.next_p;
            pix.next_p++;

        }

}

//__global__
void g_gen_kernel()
{

}
