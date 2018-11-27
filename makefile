COMPILER=nvcc -std=c++11
LINK=-L -Dcimg_use_vt100 -Dcimg_display=1 -lm -lX11 -lpthread
CUDA_link=-lcudart
targets= gcode_gen edge_detection generator

all: $(targets)

gcode_gen: gcode_gen.cu
	$(COMPILER) -c $@.cu

edge_detection: edge_detection.cu
	$(COMPILER) gcode_gen.o -c $@.cu

generator: main.cu
	$(COMPILER) main.cu edge_detection.o -o $@ $(LINK) $(CUDA_link)

clean:
	rm -rf *.o $(targets)
