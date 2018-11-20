COMPILER=g++
WERROR=-Wall -Wextra -Wfatal-errors
LINK=-L -std=c++11 -pedantic -Dcimg_use_vt100 -Dcimg_display=1 -lm -lX11 -lpthread
CUDA=-lcudart
targets= edge_detector

all: $(targets)

edge_detector: main.cpp
	$(COMPILER) main.cpp $(WERROR) $(LINK) -o $@



clean:
	rm -rf *.o $(targets)
