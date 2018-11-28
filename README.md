# parallel_gcode

To run the file use make clean, then make

# To Install the library:

sudo apt update

sudo apt install cimg-dev

# To Run graphics on external computer

ssh name@server_ip -X 

# TODO

* Clean the kernel of excess <includes>

* Get the cude implemented variation of the edge detection algorithm working

* Figure out the linking for the gcode gen file

* Maybe make the algorithm for the gcode generation not have to function on a 2d structure (got lazy)

* Error correction/detection/catching

* Figure out actual spacing of the printer to the coordinates found on the image

* Figure out spaceing and variable speed options

* Make the -c and -g option optional and only use the original image otherwise?

* Mount the sharpie to the printer.

* Parallelize the actual filter calculation (for-loop of 0-8 to thread) (kernel calling a kernel)

* Timing data in metadata (See CImg library we are using because the tutorial shows how to graph)

* Figure out dimensional scaling of the images being plotted.
