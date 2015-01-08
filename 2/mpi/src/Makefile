#Author: Mohammed Sourouri
#Date: January 21, 2014
#Updated: January 25, 2014
 
#Get OS arch
OSUPPER = $(shell uname -s 2>/dev/null)

#Get the machine
NODE_NAME=$(shell uname -n | egrep -o '^([^0-9]*)')

ifeq ("$(NODE_NAME)","lizhi")
	include simula
endif

ifeq ("$(NODE_NAME)", "alkhwarizmi")
	include simula
endif

ifeq ("$(NODE_NAME)","cvrsvc")
	include dirac
endif

# Compilers
NVCC = $(shell which nvcc)
MPICXX = $(shell which mpicxx)

# Compiler flags
CFLAGS=-m64 -O3 -march=native -Wall -fopenmp -funroll-loops
PTXFLAGS=-v
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include
MPICFLAGS=-I$(MPI_HOME)/include

# Compute flags
GENCODE_SM20 := -gencode arch=compute_20,code=sm_20
GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
GENCODE_SM35 := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

NVCCFLAGS =-O3 -m64 $(GENCODE_FLAGS) -Xcompiler -fopenmp -Xcompiler -fno-strict-aliasing -Xcompiler -funroll-loops #-Xptxas $(PTXFLAGS)

ifeq ("$(OSUPPER)","Linux")
        CUDALDFLAGS = -L${CUDA_INSTALL_PATH}/lib64 -lcudart
else
        CUDALDFLAGS = -L${CUDA_INSTALL_PATH}/lib -lcudart
endif

# Binary placement
BINDIR=../bin
SYNC=$(BINDIR)/sync
ASYNC=$(BINDIR)/async
BINARIES=$(SYNC) $(ASYNC)

# Make
all: $(BINARIES)

Util.o: Util.cu
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Kernels.o: Kernels.cu
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Sync.o: Sync.c
	 $(MPICXX) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

Async.o: Async.c
	 $(MPICXX) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<
	 
Host_MPI.o: Host_MPI.c
	$(MPICXX) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

$(SYNC): Sync.o Host_MPI.o Util.o Kernels.o
	mkdir -p $(BINDIR)
	$(MPICXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)

$(ASYNC): Async.o Host_MPI.o Util.o Kernels.o
	mkdir -p $(BINDIR)
	$(MPICXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)
			
clean:
	rm -rf *.vtk *.o $(BINARIES)
	rm -rf ../bin
