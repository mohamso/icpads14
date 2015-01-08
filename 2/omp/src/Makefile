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
CXX = $(shell which g++)

# Compiler flags
CFLAGS=-m64 -O3 -march=native -Wall -fopenmp -funroll-loops
PTXFLAGS=-v
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

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
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Kernels.o: Kernels.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
Sync.o: Sync.c
	 $(CXX) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

Async.o: Async.c
	 $(CXX) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

Host.o: Host.c
	$(CXX) $(CUDACFLAGS) $(CFLAGS) -o $@ -c $<

$(SYNC): Sync.o Host.o Util.o Kernels.o
	mkdir -p $(BINDIR)
	$(CXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)

$(ASYNC): Async.o Host.o Util.o Kernels.o
	mkdir -p $(BINDIR)
	$(CXX) -o $@ $+ $(CUDALDFLAGS) $(CFLAGS)
			
clean:
	rm -rf *.vtk *.o $(BINARIES)
	rm -rf ../bin
