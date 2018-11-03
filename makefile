NVCC=nvcc
LINK=-L /usr/local/cuda/lib
CUDA=-lcudart
targets= edge_detector

all: $(targets)

edge_detector: main.cu
	$(NVCC) $@.cu $(LINK) $(CUDA) -lm -o $@

clean:
	rm -rf *.o $(targets)
