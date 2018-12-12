# Image to G-Code Converter

The Image to G-Code converter takes an image of PNG, BMP, or JPG and uses an edge detection algorithm to output
G-Code in X and Y coordinates which is readable by a plotter. 

#### Edge Detection Algorithms:
  - *Sobel-Feldman Edge Detection*
  - *Robert's Edge Detection*
  - *Preqitt Edge Detection*

# Installation Pre-Requisites:

The program must be run on Debian based distrubutions of Linux. Any other operating
system has been untested.

#### Library Requirements for Proper Function:
  - *NVCC*
  - *CImg*

##### To Install NVCC Compiler and Dependents:
Visit [this][NVCC] website and download the latest deb and follow the commands below
or follow their instructions **(RECOMMENDED)**:
```sh
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

##### To Install CImg:

```sh
$ sudo apt-get install libx11-dev
$ sudo apt-get install cimg-dev
```

Download the project and type `make`.

# Command Usage for Tool:
```sh
./generator FILE [-i] [-g] [-d] [-o OUTPUT] [-v] [-w] [-f NUMBER]
```
| Argument | Use Case |
| ---------| -------- |
| FILE | The FILE is the name of the input file. |
| [-i] | The flag will enable the program to output the metadata of image.|
| [-g] | The flag will enable the program to process the FILE the GPU|
| [-d] | The flag enables the display the output of the edge detection. |
| [-o OUTPUT] | The flag will overwrite the default gcode output name of FILE_out.gcode | 
| [-v] (VERBOSE) | The flag enables a printout of the steps that are being performed |
| [-w] | The flag will write the image to a file when it has undergone edge detection | 
| [-f NUMBER] | The filter flag is to specify which filter the user wants to run. |
||1 -> Sobel Filter|
||2 -> Robert's Filter |
||3 -> Prewitt's Filter |
| [-h] | The helper function that is used to explain flags | 

> Example: ./generator -g -w -f 1 -t 1024


### TODO and Future Improvements

 - Optimize G-Code
 - Figure out Tiling and Memory Sharing for Speed improvements
 - More Kernels
 
----
#### NOTES:
*To run and view the display window using a client machine
when program is running on a host via SSH use the following
argument to use X11 forwarding:*
`ssh name@server_ip -X`

----
### Development and Contribution:

*This project was made for the High Performance Computing class CSE389 at
the New Mexico Institute of Mining and Technology as part of a group
project.*

Group members:
 * [Sterling Blevins][Sterling Blevins]
 * [Damon Estrada][Damon Estrada]
 * [Victor Gutierrez][Victor Gutierrez]

----

   [NVCC]: <https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork>

   [Sterling Blevins]: <https://github.com/S3Blevins>
   
   [Damon Estrada]: <https://github.com/damon-estrada>
   
   [Victor Gutierrez]: <https://github.com/vgutierrez542>
   
   
