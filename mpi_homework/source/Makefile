CC := clang++
CCFLAGS := -std=c++14 -Wall

all: matrixmul

densematgen.o: densematgen.cpp densematgen.h
	$(CC) $(CCFLAGS) -c -o $@ $<

matrixmul: matrixmul.cpp densematgen.o config.h
	$(CC) $(CCFLAGS) -o $@ $<
