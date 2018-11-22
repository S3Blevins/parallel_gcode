COMPILER=g++
WERROR=-Wall -Wextra -Wfatal-errors
LINK=-L -std=c++11 -Dcimg_use_vt100 -Dcimg_display=1 -lm -lX11 -lpthread
CUDA=-lcudart
targets= edge_detection generator

all: $(targets)

edge_detection: edge_detection.cpp
	$(COMPILER) $(WERROR) -c edge_detection.cpp

generator: main.cpp
	$(COMPILER) $(WERROR) main.cpp edge_detection.o -o $@ $(LINK)

clean:
	rm -rf *.o $(targets)
