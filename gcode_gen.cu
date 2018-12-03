// Gcode Gen class

#include "kernel.h"
#include <queue>
#include <fstream>

using namespace std;

// NOTE: everything above 50 is considered a white line
ofstream outputFile;

/**
 * Lines to print onto initial file.
 */
void gcode_prolog(void) {
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
    outputFile << "M107           ;Turn off the fan" << endl << endl;
    outputFile << ";G1 requires to extrude" << endl;
    outputFile << ";G0 does not require extrusion" << endl << endl;

    // actual gcode goes below;
    // G0 {speed} X{position} Y{position}

}

/**
 * Ending lines to print onto gcode file
 */
void gcode_epilog(double size_h, double size_w) {
    // actual gcode goes below;
    // G0 {speed} X{position} Y{position}

    outputFile << endl;
    outputFile << ";END GCODE" << endl;
    outputFile << "M104 S0        ;extruder heater off" << endl;
    outputFile << "M140 S0        ;heated bed heater off (if you have it)" << endl;
    outputFile << "G91            ;relative positioning" << endl;
    outputFile << "G28 X0 Y0      ;move X/Y to min endstops, so the head is out of the way" << endl;
    outputFile << "M84            ;steppers off" << endl;
    outputFile << "G90            ;absolute positioning" << endl;
    outputFile << ";pixel height: " << size_h << endl;
    outputFile << ";pixel width: " << size_w << endl;
}

/**
 * checks pixels adjacent to the main pixel located at x and y
 * @param image_2d      image in 2d array of format
 * @param image_visited pixel visitation flags for each pixel in 2d format
 * @param x             x position of pixel to check adjacents
 * @param y             y position of pixel to check adjacents
 */
void next_to(int **image_2d, int **image_visited, int x, int y, float size_h, float size_w) {

    int new_x;
    int new_y;
    double pos_x;
    double pos_y;

    //printf("original pixel\n");
    //printf("pixel[%d][%d] = %d\n", x, y, image_2d[x][y]);
    //printf("checking pixels...\n");

    // look at all pixels surrounding the main pixel in question
    for(int col = 0; col < 3; col++) {
        for(int row = 0; row < 3; row++) {
            new_x = x + col - 1;
            new_y = y + row - 1;
            pos_x = new_x * size_w;
            pos_y = new_y * size_h;

            //printf("checking pixel[%d][%d] = %d\n", new_x, new_y, image_visited[new_x][new_y]);
            if(image_2d[new_x][new_y] >= 50 && image_visited[new_x][new_y] == 0) {
                image_visited[new_x][new_y] = 1;
                printf("pixel[%d][%d] = %d\n", new_x, new_y, image_2d[new_x][new_y]);
                outputFile << "G0" << " F1200" << " X" << pos_x << " Y" << pos_y << endl;
                next_to(image_2d, image_visited, new_x, new_y, size_h, size_w);
            }
        }
    }
}

/**
 * processes through all pixels that have not been visited.
 * @param  image  1d vector of image
 * @param  width  [description]
 * @param  height [description]
 * @return        [description]
 */
int gcode(vector<int> image, int width, int height) {

    int **image_2d;
    image_2d = new int *[width];
    int **image_visited;
    image_visited = new int *[width];
    double size_w = (double)190/width;
    double size_h = (double)190/height;

    cout << "width: " << size_w << endl;
    cout << "height: " << size_h << endl;

    gcode_prolog();

    // rebuild the image in 2d format
    // NOTE: we could probably format in 1d but I didnt want to spend much
    // time on this in case somebody had another idea
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
                outputFile << "G0 F1200 Z0.03\t\t\t;move pencil down" << endl;

                // recursive call once a grey/white pixel has been found
                // and follow up with any pixels which are grey/white
                // immediately next to that
                next_to(image_2d, image_visited, x, y, size_h, size_w);

                outputFile << "G0 F1200 Z15.0\t\t\t;move pencil up" << endl;
             }
         }
     }

     gcode_epilog(size_h, size_w);

    return 0;
}

/**
 * gcode generator wrapper function.
 * @param img         1d vector of image pixel contents
 * @param width       the width of the image
 * @param height      the height of the image
 * @param output_name output of name to be written
 */
void g_gen(vector<int> img, int width, int height, string output_name) {

    // add gcode file extension
    output_name.append(".gcode");

    // open file
    // NOTE: we should probably throw in a try-catch and figure out write flags
    outputFile.open(output_name);

    // call the function
    gcode(img, width, height);

    // close file
    //outputFile.close(output_name);

}
