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

#define DEBUG
#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////
// A method for checking error in CUDA calls
////////////////////////////////////////////////////////////////////////////////
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, 
			cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

////////////////////////////////////////////////////////////////////////////////
// Program Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	int Nx, Ny, Nz, max_iters;
	int blockX, blockY, blockZ;

	if (argc == 8) {
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
		printf("Usage: %s nx ny nz i block_x block_y block_z number_of_threads\n", 
			argv[0]);
		exit(1);
	}

	// Get the number of GPUS
	int number_of_devices;
	checkCuda(cudaGetDeviceCount(&number_of_devices));
  
  if (number_of_devices < 2) {
  	printf("Less than two devices were found.\n");
  	printf("Exiting...\n");

  	return -1;
  }

	// Decompose along the Z-axis
	int _Nz = Nz/number_of_devices;

	// Define constants
	const _DOUBLE_ L = 1.0;
	const _DOUBLE_ h = L/(Nx+1);
	const _DOUBLE_ dt = h*h/6.0;
	const _DOUBLE_ beta = dt/(h*h);
	const _DOUBLE_ c0 = beta;
	const _DOUBLE_ c1 = (1-6*beta);

	// Check if ECC is turned on
	ECCCheck(number_of_devices);

	// Set the number of OpenMP threads
	omp_set_num_threads(4);

	#pragma omp parallel
	{
		unsigned int tid = omp_get_num_threads();

		#pragma omp single
		{
			printf("Number of OpenMP threads: %d\n", tid);
		}
	}

  // CPU memory operations
  int dt_size = sizeof(_DOUBLE_);

	_DOUBLE_ *u_new, *u_old;

	u_new = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2));
	u_old = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2));

	init(u_old, u_new, h, Nx, Ny, Nz);

	// Allocate and generate arrays on the host
	size_t pitch_bytes;
	size_t pitch_gc_bytes;

	_DOUBLE_ *h_Unew, *h_Uold;
	_DOUBLE_ *h_s_Uolds[number_of_devices], *h_s_Unews[number_of_devices];
	_DOUBLE_ *left_send_buffer[number_of_devices], *left_receive_buffer[number_of_devices];
	_DOUBLE_ *right_send_buffer[number_of_devices], *right_receive_buffer[number_of_devices];

	h_Unew = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2));
	h_Uold = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(Nz+2));

	init(h_Uold, h_Unew, h, Nx, Ny, Nz);

	#pragma omp parallel
	{
		unsigned int tid = omp_get_thread_num();

		h_s_Unews[tid] = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_Nz+2));
		h_s_Uolds[tid] = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_Nz+2));

		right_send_buffer[tid] = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
		left_send_buffer[tid] = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
		right_receive_buffer[tid] = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));
		left_receive_buffer[tid] = (_DOUBLE_ *)malloc(sizeof(_DOUBLE_)*(Nx+2)*(Ny+2)*(_GC_DEPTH));

		checkCuda(cudaHostAlloc((void**)&h_s_Unews[tid], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
		checkCuda(cudaHostAlloc((void**)&h_s_Uolds[tid], dt_size*(Nx+2)*(Ny+2)*(_Nz+2), cudaHostAllocPortable));
		checkCuda(cudaHostAlloc((void**)&right_send_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
		checkCuda(cudaHostAlloc((void**)&left_send_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
		checkCuda(cudaHostAlloc((void**)&right_receive_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));
		checkCuda(cudaHostAlloc((void**)&left_receive_buffer[tid], dt_size*(Nx+2)*(Ny+2)*(_GC_DEPTH), cudaHostAllocPortable));

		init_subdomain(h_s_Uolds[tid], h_Uold, Nx, Ny, _Nz, tid);
	}

	// GPU memory operations
	_DOUBLE_ *d_s_Unews[number_of_devices], *d_s_Uolds[number_of_devices];
	_DOUBLE_ *d_right_send_buffer[number_of_devices], *d_left_send_buffer[number_of_devices];
	_DOUBLE_ *d_right_receive_buffer[number_of_devices], *d_left_receive_buffer[number_of_devices];

#pragma omp parallel
{
	unsigned int tid = omp_get_thread_num();

	if (tid < 2) {

		checkCuda(cudaSetDevice(tid));
		CopyToConstantMemory(c0, c1);

		checkCuda(cudaMallocPitch((void**)&d_s_Uolds[tid], &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
		checkCuda(cudaMallocPitch((void**)&d_s_Unews[tid], &pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2)));
		checkCuda(cudaMallocPitch((void**)&d_left_receive_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
		checkCuda(cudaMallocPitch((void**)&d_right_receive_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
		checkCuda(cudaMallocPitch((void**)&d_left_send_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
		checkCuda(cudaMallocPitch((void**)&d_right_send_buffer[tid], &pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH)));
	}
}

// GPU stream operations
cudaStream_t compute_stream_2, compute_stream_3;
cudaStream_t right_send_stream, left_send_stream;
cudaStream_t right_receive_stream, left_receive_stream;

#pragma omp parallel
{
	unsigned int tid = omp_get_thread_num();

	if (tid == 0) {
			checkCuda(cudaSetDevice(0));
			checkCuda(cudaStreamCreate(&right_send_stream));
			checkCuda(cudaStreamCreate(&right_receive_stream));
	}
	if (tid == 2) {
			checkCuda(cudaSetDevice(0));
			checkCuda(cudaStreamCreate(&compute_stream_2));
	}
	if (tid == 1) {
			checkCuda(cudaSetDevice(1));
			checkCuda(cudaStreamCreate(&left_send_stream));
			checkCuda(cudaStreamCreate(&left_receive_stream));
	}
	if (tid == 3) {
			checkCuda(cudaSetDevice(1));
			checkCuda(cudaStreamCreate(&compute_stream_3));
	}
}

	// Copy data from host to the device
	double HtD_timer = 0.;
	HtD_timer -= omp_get_wtime();
#pragma omp parallel
{
	unsigned int tid = omp_get_thread_num();

	if (tid < 2) {
		checkCuda(cudaSetDevice(tid));
		checkCuda(cudaMemcpy2D(d_s_Uolds[tid], pitch_bytes, h_s_Uolds[tid], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy2D(d_s_Unews[tid], pitch_bytes, h_s_Unews[tid], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_Nz+2)), cudaMemcpyHostToDevice));
	}
}
	HtD_timer += omp_get_wtime();

	int pitch = pitch_bytes/dt_size;
	int gc_pitch = pitch_gc_bytes/dt_size;

  // GPU kernel launch parameters
	dim3 threads_per_block(blockX, blockY, blockZ);
	unsigned int blocksInX = getBlock(Nx, blockX);
	unsigned int blocksInY = getBlock(Ny, blockY);
	unsigned int blocksInZ = getBlock(_Nz-2, k_loop);
	dim3 thread_blocks(blocksInX, blocksInY, blocksInZ);
	dim3 thread_blocks_halo(blocksInX, blocksInY);

	unsigned int ghost_width = 1;

	double compute_timer = 0.;
  compute_timer -= omp_get_wtime();

	#pragma omp parallel
	{
		unsigned int tid = omp_get_thread_num();

		for(int iterations = 0; iterations < max_iters; iterations++) {

			// Make sure that all threads start at the same time
			#pragma omp barrier

			// Compute right boundary data on device 0
			if (tid == 0) {
				int kstart = (_Nz+1)-ghost_width;
				int kstop = _Nz+1;

				checkCuda(cudaSetDevice(0));

				ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, 
				right_send_stream, d_s_Unews[0], d_s_Uolds[0], pitch, Nx, Ny, _Nz, kstart, kstop);

				CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, 
				right_send_stream, d_s_Unews[0], d_right_send_buffer[0], Nx, Ny, _Nz, pitch, gc_pitch, 0);

				checkCuda(cudaMemcpy2DAsync(right_send_buffer[0], dt_size*(Nx+2), 
				d_right_send_buffer[0], pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH), cudaMemcpyDeviceToHost, right_send_stream));
		}

		// Compute left boundary data on device 1
		if (tid == 1) {
			int kstart = 1;
		  int kstop = 1+ghost_width;

			checkCuda(cudaSetDevice(1));

			ComputeInnerPointsAsync(thread_blocks_halo, threads_per_block, 
				left_send_stream, d_s_Unews[1], d_s_Uolds[1], pitch, Nx, Ny, _Nz, kstart, kstop);

			CopyBoundaryRegionToGhostCellAsync(thread_blocks_halo, threads_per_block, 
				left_send_stream, d_s_Unews[1], d_left_send_buffer[1], Nx, Ny, _Nz, pitch, gc_pitch, 1);

			checkCuda(cudaMemcpy2DAsync(left_send_buffer[1], dt_size*(Nx+2), 
				d_left_send_buffer[1], pitch_gc_bytes, dt_size*(Nx+2), (Ny+2)*(_GC_DEPTH), cudaMemcpyDeviceToHost, left_send_stream));
		}

		// Compute inner nodes for device 0
		if (tid == 2) {
			int kstart = 1;
			int kstop = (_Nz+1)-ghost_width;
			
			checkCuda(cudaSetDevice(0));

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, 
				compute_stream_2, d_s_Unews[0], d_s_Uolds[0], pitch, Nx, Ny, _Nz, kstart, kstop);
		}
		
		// Compute inner nodes for device 1
		if (tid == 3) {
			int kstart = 1+ghost_width;
			int kstop = _Nz+1;
			
			checkCuda(cudaSetDevice(1));

			ComputeInnerPointsAsync(thread_blocks, threads_per_block, 
				compute_stream_3, d_s_Unews[1], d_s_Uolds[1], pitch, Nx, Ny, _Nz, kstart, kstop);
		}

		#pragma omp barrier

		if (tid == 1) {
			while (true) {
				cudaSetDevice(0);
				if (cudaStreamQuery(right_send_stream) == cudaSuccess) {
					cudaSetDevice(1);

					checkCuda(cudaMemcpy2DAsync(d_left_receive_buffer[1], pitch_gc_bytes, 
					right_send_buffer[0], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_GC_DEPTH)), cudaMemcpyHostToDevice, left_receive_stream));

					CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, 
						left_receive_stream, d_s_Unews[1], d_left_receive_buffer[1], Nx, Ny, _Nz, pitch, gc_pitch, 1);
					break;
				}
			}
		}

		if (tid == 0) {
			while (true) {
				cudaSetDevice(1);
				if (cudaStreamQuery(left_send_stream) == cudaSuccess) {
					cudaSetDevice(0);

					checkCuda(cudaMemcpy2DAsync(d_right_receive_buffer[0], pitch_gc_bytes,
						left_send_buffer[1], dt_size*(Nx+2), dt_size*(Nx+2), ((Ny+2)*(_GC_DEPTH)), cudaMemcpyHostToDevice, right_receive_stream));

					CopyGhostCellToBoundaryRegionAsync(thread_blocks_halo, threads_per_block, 
						right_receive_stream, d_s_Unews[0], d_right_receive_buffer[0], Nx, Ny, _Nz, pitch, gc_pitch, 0);
					break;
				}
			}
		}

		// Swap pointers on the host
		#pragma omp barrier
		if (tid < 2) {
			checkCuda(cudaDeviceSynchronize());
			swap(_DOUBLE_*, d_s_Unews[tid], d_s_Uolds[tid]);
		}
	}
}

	compute_timer += omp_get_wtime();

  // Copy data from device to host
	double DtH_timer = 0;
  DtH_timer -= omp_get_wtime();
	#pragma omp parallel
	{
		unsigned int tid = omp_get_thread_num();

		if (tid < 2) {
			checkCuda(cudaSetDevice(tid));
			checkCuda(cudaMemcpy2D(h_s_Uolds[tid], dt_size*(Nx+2), d_s_Uolds[tid], pitch_bytes, dt_size*(Nx+2), (Ny+2)*(_Nz+2), cudaMemcpyDeviceToHost));
		}
	}
	DtH_timer += omp_get_wtime();

	// Merge sub-domains into a one big domain
	#pragma omp parallel
	{
		unsigned int tid = omp_get_thread_num();
		if (tid < 2) {
			merge_domains(h_s_Uolds[tid], h_Uold, Nx, Ny, _Nz, tid);
		}
	}

   	// Calculate on host
#if defined(DEBUG) || defined(_DEBUG)
	cpu_heat3D(u_new, u_old, c0, c1, max_iters, Nx, Ny, Nz);
#endif

    float gflops = CalcGflops(compute_timer, max_iters, Nx, Ny, Nz);
    PrintSummary("3D Heat (7-pt)", "Plane sweeping", compute_timer, HtD_timer, DtH_timer, gflops, max_iters, Nx);

    _DOUBLE_ t = max_iters * dt;
    CalcError(h_Uold, u_old, t, h, Nx, Ny, Nz);

#if defined(DEBUG) || defined(_DEBUG)
    //exportToVTK(h_Uold, h, "heat3D.vtk", Nx, Ny, Nz);
#endif

	#pragma omp parallel
	{
		unsigned int tid = omp_get_thread_num();

		if (tid < 2)
		{
			checkCuda(cudaSetDevice(tid));
			checkCuda(cudaFree(d_s_Unews[tid]));
	    checkCuda(cudaFree(d_s_Uolds[tid]));
	    checkCuda(cudaFree(d_right_send_buffer[tid]));
	    checkCuda(cudaFree(d_left_send_buffer[tid]));
	    checkCuda(cudaFree(d_right_receive_buffer[tid]));
	    checkCuda(cudaFree(d_left_receive_buffer[tid]));
	    checkCuda(cudaFreeHost(h_s_Unews[tid]));
	    checkCuda(cudaFreeHost(h_s_Uolds[tid]));
	    checkCuda(cudaFreeHost(left_send_buffer[tid]));
	    checkCuda(cudaFreeHost(right_send_buffer[tid]));
	    checkCuda(cudaFreeHost(left_receive_buffer[tid]));
	    checkCuda(cudaFreeHost(right_receive_buffer[tid]));
			checkCuda(cudaDeviceReset());
  }

  if (tid == 0) {
		checkCuda(cudaSetDevice(0));
		checkCuda(cudaStreamCreate(&right_send_stream));
		checkCuda(cudaStreamCreate(&right_receive_stream));
	}

	if (tid == 2) {
		checkCuda(cudaSetDevice(0));
		checkCuda(cudaStreamCreate(&compute_stream_2));
	}
	
	if (tid == 1) {
		checkCuda(cudaSetDevice(1));
		checkCuda(cudaStreamCreate(&left_send_stream));
		checkCuda(cudaStreamCreate(&left_receive_stream));
	}

	if (tid == 3) {
		checkCuda(cudaSetDevice(1));
		checkCuda(cudaStreamCreate(&compute_stream_3));
	}
}

  free(u_old);
  free(u_new);

	return 0;
}
