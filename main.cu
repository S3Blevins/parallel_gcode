/**
 * @file main.cu
 *
 * @author Sterling Blevins
 *
 * @date 12/11/18
 *
 * @assignment: HPC Project
 *
 * @brief Project detects edges of image using GPU or CPU and converts the
 * edges into G-Code which is used by a plotter
 *
 * @bugs Seems to function!
 */
#include "kernel.h" // Contains function prototypes and libs
#define BUFF 4096

using namespace std;

void helper(void);

/**
 * Main function which processes the user input
 * @param  argc number of arguments
 * @param  argv arguments as collection of strings
 * @return      0
 */
int main(int argc, char *argv[]) {

    // if no file has been mentioned, throw an error.
    if(argv[1] == NULL) {
        printf("ERROR: Please use [-h] for proper runtime info\n");
        helper();
        exit(0);
    }

    // Check if file exists, exit else
    ifstream exists(argv[1], ios::binary);
    if(!exists) {
        printf("Image file has not been located. \nPlease use an appropriate image.\n");
        exit(0);
    }

    // | bit |      type      |            values
    // |  0  | processor flag | 0 CPU          / 1 GPU
    // |  1  | display flag   | 0 display off  / 1 display on
    // |  2  | verbose mode   | 0 verbose off  / 1 verbose on
    // |  3  | metadata flag  | 0 meta off     / 1 meta on
    // |  4  | write image    | 0 write off    / 1 write on
    char flags = 0x0;
    int threshold = 2048;

    // default flags written
    int default_out_flag = 0;
    int filter = 1;
    int test_count = 1;

    // string allocations for input and output names
    string input_name = argv[1];
    string output_name = input_name;

    int threshold_changed = 0;

    int c;              // switch case variable
    while((c = getopt(argc, argv, "icgdo:f:r:vwt:h")) != -1) {
        switch(c) {
            case 'i':   // run with metadata output
                flags |= 0x8;
                break;
            case 'g':   // run edge_detection on GPU (CANNOT BE RUN WITH '-c' FLAG)
                flags |= 0x1;
                break;
            case 'd':   // display the edge detection filter applied to image
                flags |= 0x2;
                break;
            case 'o':   // change gcode output name
                output_name = optarg;
                default_out_flag = 1;
                break;
            case 'f':   // filter type
                filter = atoi(optarg); //change the filter choise to an int
                break;
            case 'v':   // enable verbose mode [FOR DEBUGGING]
                flags |= 0x4;
                break;
            case 'w':   // enables writing of edge detection filter to a file
                flags |= 0x10;
                break;
            case 'r':   // run the kernel x amount
                //initialized by default to 1
                test_count = atoi(optarg); // How many times to run a certain filter.
                break;
            case 't':   // filter threshold (HIDDEN FLAG -> DEFAULT is 2048)
                try {
                    threshold = stoi(optarg);
                    threshold_changed = 1;
                    //printf("threshold %d\n", threshold);
                } catch (std::invalid_argument&e) {
                    printf("Please use a proper threshold value.\n");
                    exit(0);
                }
                break;
            case 'h':   // helper function (falls through to default)
            default:    // defaults to usage again
                helper();
                break;
            }
    }

    // set default filter thresholds
    if((filter == 2) && (threshold_changed == 0)) {
        threshold = 1024;
    } else if((filter == 3) && (threshold_changed == 0)) {
        threshold = 512;
    }

    // isolate the string if the flag has not been selected.
    if(!default_out_flag) {
        string delimiter = ".";
        output_name = output_name.substr(0, output_name.find(delimiter));
        output_name.append("_out");
    }

    // call the wrapper function which compiles according to the flags
    for (int t = 0; t < test_count; t++) {
        edge_detection_wrapper(flags, input_name, output_name, threshold, filter);
    }

    return 0;
}

/**
 * The helper function outputs command and flag usage.
 */
void helper(void) {
    printf("\nCOMMAND USAGE FOR TOOL:\n\n");
    printf("\t./generator FILE [-i] [-g] [-d] [-o OUTPUT] [-v] [-w] [-f NUMBER]\n\n");

    printf("\tFILE (REQUIRED)\n");
    printf("\tThe FILE is the name of the input file.\n\n");

    printf("\t[-i]\n");
    printf("\tThe flag will enable the program to output the metadata of image\n\n");

    printf("\t[-g] (REQUIRED or [-c])\n");
    printf("\tThe flag will enable the program to process the FILE the GPU\n");
    printf("\tNOTE: This flag cannot be used alongside the '-c' flag.\n\n");

    printf("\t[-d]\n");
    printf("\tThe flag enables the display the output of the edge detection.\n");
    printf("\tNOTE: X11 forwarding required and -X flag when SSH'ing.\n\n");

    printf("\t[-o OUTPUT]\n");
    printf("\tThe flag will overwrite the default gcode output name of FILE_out.gcode\n");
    printf("\tto 'OUTPUT.gcode'.\n\n");

    printf("\t[-v] (VERBOSE)\n");
    printf("\tThe flag enables a printout of the steps that are being performed\n");
    printf("\tby the program as they are executed.\n\n");

    printf("\t[-w]\n");
    printf("\tThe flag will write the image to a file when it has undergone the\n");
    printf("\tedge detection.\n");
    printf("\tNOTE: Will assume the name of the OUTPUT with image file extension.\n\n");

    printf("\t[-h]\n");
    printf("\tThe helper function that is used to explain flags and the functionality\n");
    printf("\tof the program.\n\n");

    printf("\t[-f]\n");
    printf("\tThe filter flag is to specify which filter the user wants to run.\n");
    printf("\tThe following filters are supported and denoated by: \n\n");
    printf("\tSobel Edge detector: 1\n");
    printf("\tRobert's Edge detector: 2\n");
    printf("\tPrewitt Edge dector: 3\n");

    printf("\tExample: ./generator -g -w -f 1 -t 1024 \n");
}
