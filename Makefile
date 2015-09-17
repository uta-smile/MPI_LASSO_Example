GSLROOT=/usr/local
#GSLROOT=/usr
# use this if on 64-bit machine with 64-bit GSL libraries
#ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386

#To use the GSL library, compile the source code with the option:
#   -I$TACC_GSL_INC -I$TACC_GSL_INC/gsl
#and add the following commands to the link step: 
# -L$TACC_GSL_LIB -lgsl -lgslcblas


MPICC=mpicc
CC=gcc
# ---- For local machine
#CFLAGS=-Wall -std=c99 -arch $(ARCH) -I$(GSLROOT)/include
#LDFLAGS=-L$(GSLROOT)/lib -lgsl -lgslcblas -lm
# ---- For running on TACC
#CFLAGS=-Wall -g -std=c99 -I$(TACC_GSL_INC) -I$(TACC_GSL_INC)/gsl
#LDFLAGS=-L$(TACC_GSL_LIB) -lgsl -lgslcblas -lm
# ---- ugly version for all 
CFLAGS=-Wall -std=c99 -I$(GSLROOT)/include -I$(TACC_GSL_INC) -I$(TACC_GSL_INC)/gsl 
LDFLAGS=-L$(TACC_GSL_LIB) -L$(GSLROOT)/lib -lgsl -lgslcblas -lm

all: lasso 

lasso: lasso.o mmio.o
	$(MPICC) $(CFLAGS) lasso.o mmio.o -o lasso $(LDFLAGS)

lasso.o: lasso.c mmio.o
	$(MPICC) $(CFLAGS) -c lasso.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o lasso 

run:
	mpirun -np 4 ./lasso

runpy:
	mpirun -np 4 python ./lasso.py

runlua:
	mpirun -np 4 th ./lasso.lua











