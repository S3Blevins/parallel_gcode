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
    int count;
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

    pix.value = &matrix[0][0] - 1;
    pix.next_p = pix.value + 1;
    pix.count = 0;

    // traversing the matrix
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
        {
            pix.value = pix.next_p;
            pix.next_p++;

            if(!*pix.value)
                continue;

            if(!*pix.next_p)
                continue;

            pix.count = pix.count + 1;
            printf("value: (%d, %d), next_p: (%d), count: %d \n", row,col, *pix.next_p, pix.count);
        }

}

//__global__
void g_gen_kernel()
{

}
