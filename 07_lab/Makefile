#
# A template for the 2016 MPI lab at the University of Warsaw.
# Copyright (C) 2016, Konrad Iwanicki.
# Further modifications by Krzysztof Rzadca 2018
#

CC          := gcc
MPI			    := cc
CFLAGS      := -O3 -c -Wall
LFLAGS      := -O3
# Add new targets below:
ALL         := hello-world-seq.exe benchmark-latency.exe

all : $(ALL)


%.exe : %.c
	$(MPI) -o $@ $<

clean :
	rm -f *.o *.out *.err $(ALL)

