CC=g++
CFLAGS= -g -Wall --debug

TARGETS = main
FILES = main.cpp gcode_gen.cpp edge_detection.cpp kernel.h

all: $(TARGETS)

main: $(FILES)
	$(CC) $(CFLAGS) gcode_gen.cpp edge_detection.cpp $@.cpp -o $@

clean:
	rm -f *.o $(TARGETS);
