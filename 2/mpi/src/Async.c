/* Filename: Async.c
 * Author: Mohammed Sourouri <mohamso@simula.no>
 * 
 * Asnchronous state-of-the-art Multi-GPU code where the number of MPI processes
 * spawned equals the number of GPUs. All memory transfers are asynchronous. 
 * Non-blocking MPI calls are used. This code corresponds to "MPI" results in 
 * Figure-9 in the paper.
 *
 * 
 * Copyright 2014 Mohammed Sourouri
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Sync.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////
// A method for checking error in CUDA calls
////////////////////////////////////////////////////////////////////////////////
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

///////////////////////
// Main program entry
///////////////////////
int main(int argc, char** argv)
{
	unsigned int max_iters, Nx, Ny, Nz, blockX, blockY, blockZ;
	int rank, numberOfProcesses;

	if (argc == 8)
	{
		Nx = atoi(argv[1]);
		Ny = atoi(argv[2]);
		Nz = atoi(argv[3]);
		max_iters = atoi(argv[4]);
		blockX = atoi(argv[5]);
		blockY = atoi(argv[6]);
		blockZ = atoi(argv[7]);
	}
	else
	{
		printf("Usage: %s nx ny nz i block_x block_y block_z\n", argv[0]);
		exit(1);
	}

  InitializeMPI(&argc, &argv, &rank, &numberOfProcesses);
  AssignDevices(rank);
  ECCCheck(rank);

	// Define constants
	const _DOUBLE_ L = 1.0;
	const _DOUBLE_ h = L/(Nx+1);
	const _DOUBLE_ dt = h*h/6.0;
	const _DOUBLE_ beta = dt/(h*h);
	const _DOUBLE_ c0 = beta;
	const _DOUBLE_ c1 = (1-6*beta);

	// Copy constants to Constant Memory on the GPUs
	CopyToConstantMemory(c0, c1);

	// Decompose along the z-axis
	const int _Nz = Nz/numberOfProcesses;
  const int dt_size = sizeof(_DOUBLE_);

    // Host memory allocations
    _DOUBLE_ *u_new, *u_old;
    _DOUBLE_ *h_Uold;

    u_new = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2));
    u_old = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2));

    if (rank == 0)
    {
    	h_Uold = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2)); 
    }

    init(u_old, u_new, h, Nx, Ny, Nz);

    // Allocate and generate host subdomains
    _DOUBLE_ *h_s_Uolds, *h_s_Unews, *h_s_rbuf[numberOfProcesses];
    _DOUBLE_ *left_send_buffer, *left_receive_buffer;
    _DOUBLE_ *right_send_buffer, *right_receive_buffer;

    h_s_Unews = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_Nz+2));
    h_s_Uolds = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_Nz+2));

#if defined(DEBUG) || defined(_DEBUG)
  if (rank == 0)
  {
    for (int i = 0; i < numberOfProcesses; i++)
    {
        h_s_rbuf[i] = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_Nz+2));
        checkCuda(cudaHostAlloc((void**)&h_s_rbuf[i], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
    }
  }
#endif

    right_send_buffer = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
    left_send_buffer = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
    right_receive_buffer = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
    left_receive_buffer = (_DOUBLE_*)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));

    checkCuda(cudaHostAlloc((void**)&h_s_Unews, dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&h_s_Uolds, dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));

    checkCuda(cudaHostAlloc((void**)&right_send_buffer, dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&left_send_buffer, dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&right_receive_buffer, dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
    checkCuda(cudaHostAlloc((void**)&left_receive_buffer, dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));

    init_subdomain(h_s_Uolds, u_old, Nx, Ny, _Nz, rank);

	// GPU stream operations
	cudaStream_t compute_stream;
	cudaStream_t data_stream;

	checkCuda(cudaStreamCreate(&compute_stream));
	checkCuda(cudaStreamCreate(&data_stream));

	// GPU Memory Operations
	size_t pitch_bytes, pitch_gc_bytes;

  _DOUBLE_ *d_s_Unews, *d_s_Uolds;
  _DOUBLE_ *d_right_send_buffer, *d_left_send_buffer;
  _DOUBLE_ *d_right_receive_buffer, *d_left_receive_buffer;

  checkCuda(cudaMallocPitch((void**)&d_s_Uolds, &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
  checkCuda(cudaMallocPitch((void**)&d_s_Unews, &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));

  checkCuda(cudaMallocPitch((void**)&d_left_send_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
  checkCuda(cudaMallocPitch((void**)&d_left_receive_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
  checkCuda(cudaMallocPitch((void**)&d_right_send_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
  checkCuda(cudaMallocPitch((void**)&d_right_receive_buffer, &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));

	// Copy subdomains from host to device and get walltime
	double HtD_timer = 0.;

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	HtD_timer -= MPI_Wtime();
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  checkCuda(cudaMemcpy2D(d_s_Uolds, pitch_bytes, h_s_Uolds, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyDefault));
  checkCuda(cudaMemcpy2D(d_s_Unews, pitch_bytes, h_s_Unews, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyDefault));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	HtD_timer += MPI_Wtime();
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	unsigned int ghost_width = 1;

	int pitch = pitch_bytes/dt_size;
	int gc_pitch = pitch_gc_bytes/dt_size;

  // GPU kernel launch parameters
	dim3 threads_per_block(blockX, blockY, blockZ);
	unsigned int blocksInX = getBlock(Nx, blockX);
	unsigned int blocksInY = getBlock(Ny, blockY);
	unsigned int blocksInZ = getBlock(_Nz-2, k_loop);

	dim3 thread_blocks(blocksInX, blocksInY, blocksInZ);
	dim3 thread_blocks_halo(blocksInX, blocksInY);

	//MPI_Status status;
	MPI_Status status[numberOfProcesses];
	MPI_Request gather_send_request[numberOfProcesses];
	MPI_Request right_send_request[numberOfProcesses], left_send_request[numberOfProcesses];
	MPI_Request right_receive_request[numberOfProcesses], left_receive_request[numberOfProcesses];

	double compute_timer = 0.;

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  compute_timer -= MPI_Wtime();
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	for(unsigned int iterations = 0; iterations < max_iters; iterations++)
	{
		// Compute right boundary data on device 0
		if (rank == 0) {
      int kstart = (_Nz+1)-ghost_width;
	    int kstop = _Nz+1;

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, data_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, data_stream, d_s_Unews, d_right_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 0);
			
			checkCuda(cudaMemcpy2DAsync(right_send_buffer, dt_size*(Nx+2), d_right_send_buffer, pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH), cudaMemcpyDefault, data_stream));
			checkCuda(cudaStreamSynchronize(data_stream));

			MPI_CHECK(MPI_Isend(right_send_buffer, (Nx+2)*(Ny+2)*(_GC_DEPTH), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &right_send_request[rank]));
		}
		else
		{
			int kstart = 1;
			int kstop = 1+ghost_width;

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, data_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, data_stream, d_s_Unews, d_left_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 1);
			
			checkCuda(cudaMemcpy2DAsync(left_send_buffer, dt_size*(Nx+2), d_left_send_buffer, pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH), cudaMemcpyDefault, data_stream));
			checkCuda(cudaStreamSynchronize(data_stream));

			MPI_CHECK(MPI_Isend(left_send_buffer, (Nx+2)*(Ny+2)*(_GC_DEPTH), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &left_send_request[rank]));
		}

		// Compute inner nodes for device 0
		if (rank == 0) {
			int kstart = 1;
			int kstop = (_Nz+1)-ghost_width;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
		}
		// Compute inner nodes for device 1
		else
		{
			int kstart = 1+ghost_width;
			int kstop = _Nz+1;

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, compute_stream, d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		// Receive data from device 1
		if (rank == 0) {
			MPI_CHECK(MPI_Irecv(left_send_buffer, (Nx+2)*(Ny+2)*(_GC_DEPTH), MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &right_receive_request[rank]));
		}
		else
		{
			MPI_CHECK(MPI_Irecv(right_send_buffer, (Nx+2)*(Ny+2)*(_GC_DEPTH), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &left_receive_request[rank]));
		}

		if (rank == 0) {
      MPI_CHECK(MPI_Wait(&right_receive_request[rank], &status[rank]));

			checkCuda(cudaMemcpy2DAsync(d_right_receive_buffer, pitch_gc_bytes, left_send_buffer, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_GC_DEPTH)), cudaMemcpyDefault, data_stream));
			CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, data_stream, d_s_Unews, d_right_receive_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 0);
		}
		else
		{
			MPI_CHECK(MPI_Wait(&left_receive_request[rank], &status[rank]));

			checkCuda(cudaMemcpy2DAsync(d_left_receive_buffer, pitch_gc_bytes, right_send_buffer, dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_GC_DEPTH)), cudaMemcpyDefault, data_stream));
			CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, data_stream, d_s_Unews, d_left_receive_buffer, Nx, Ny, _Nz, pitch, gc_pitch, 1);
		}

		if (rank == 0)
		{
			MPI_CHECK(MPI_Wait(&right_send_request[rank], MPI_STATUS_IGNORE));
		}
		else
		{
			MPI_CHECK(MPI_Wait(&left_send_request[rank], MPI_STATUS_IGNORE));
		}

		// Swap pointers on the host
		checkCuda(cudaDeviceSynchronize());
		swap(_DOUBLE_*, d_s_Unews, d_s_Uolds);
	}

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	compute_timer += MPI_Wtime();
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	// Copy data from device to host
	double DtH_timer = 0;

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  DtH_timer -= MPI_Wtime();
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	checkCuda(cudaMemcpy2D(h_s_Uolds, dt_size*(Nx+2), d_s_Uolds, pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2), cudaMemcpyDefault));

	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	DtH_timer += MPI_Wtime();
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Gather results from subdomains
  MPI_CHECK(MPI_Isend(h_s_Uolds, (Nx+2)*(Ny+2)*(_Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &gather_send_request[rank]));

	if (rank == 0)
	{
		for (int i = 0; i < numberOfProcesses; i++)
		{
			MPI_CHECK(MPI_Recv(h_s_rbuf[i], (Nx+2)*(Ny+2)*(_Nz+2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status[rank]));
			merge_domains(h_s_rbuf[i], h_Uold, Nx, Ny, _Nz, i);
		}
	}

	// Calculate on host
#if defined(DEBUG) || defined(_DEBUG)
	if (rank == 0)
	{
		cpu_heat3D(u_new, u_old, c0, c1, max_iters, Nx, Ny, Nz);
	}
#endif

	if (rank == 0)
	{
		float gflops = CalcGflops(compute_timer, max_iters, Nx, Ny, Nz);
		PrintSummary("3D Heat (7-pt)", "Plane sweeping", compute_timer, HtD_timer, DtH_timer, gflops, max_iters, Nx);

		_DOUBLE_ t = max_iters * dt;
		CalcError(h_Uold, u_old, t, h, Nx, Ny, Nz);
	}

	Finalize();

  // Free device memory
  checkCuda(cudaFree(d_s_Unews));
  checkCuda(cudaFree(d_s_Uolds));
  checkCuda(cudaFree(d_right_send_buffer));
  checkCuda(cudaFree(d_left_send_buffer));
  checkCuda(cudaFree(d_right_receive_buffer));
  checkCuda(cudaFree(d_left_receive_buffer));

  // Free host memory
  checkCuda(cudaFreeHost(h_s_Unews));
  checkCuda(cudaFreeHost(h_s_Uolds));

#if defined(DEBUG) || defined(_DEBUG)
  if (rank == 0)
  {
  	for (int i = 0; i < numberOfProcesses; i++)
  	{
  		checkCuda(cudaFreeHost(h_s_rbuf[i]));
  	}

    free(h_Uold);
  }
#endif

  checkCuda(cudaFreeHost(left_send_buffer));
  checkCuda(cudaFreeHost(left_receive_buffer));
  checkCuda(cudaFreeHost(right_send_buffer));
  checkCuda(cudaFreeHost(right_receive_buffer));

  checkCuda(cudaDeviceReset());

  free(u_old);
  free(u_new);

  return 0;
}
