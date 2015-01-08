#include "Sync.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

__constant__ _DOUBLE_ d_c0;
__constant__ _DOUBLE_ d_c1;

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

////////////////////////////////////////////////////////////////////////////////
// Kernel for computing halos
////////////////////////////////////////////////////////////////////////////////
__global__ void copy_br_to_gc(_DOUBLE_ * __restrict u_news, _DOUBLE_ * __restrict gc_unews, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int pitch, const unsigned int gc_pitch, const unsigned int p)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int k_off = pitch*(Ny+2);

	unsigned int k = _Nz + p*(1-_Nz);

	unsigned int idx3d = i + j * pitch + k * k_off;
	unsigned int idx2d = j * gc_pitch + i;

	if( i < Nx+2 && j < Ny+2 && k < _Nz+2 )
	{
		gc_unews[idx2d] = u_news[idx3d];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Kernel for computing halos
////////////////////////////////////////////////////////////////////////////////
__global__ void copy_gc_to_br(_DOUBLE_ * __restrict u_news, _DOUBLE_ * __restrict gc_unews, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int pitch, const unsigned int gc_pitch, const unsigned int p)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int k = (1-p)*(_Nz+1);

	unsigned int idx2d = j * gc_pitch + i;

	unsigned int idx3d = i + j * pitch + k*(Ny+2)*pitch;

	if( i < Nx+2 && j < Ny+2 && k < _Nz+2 )
	{
		u_news[idx3d] = gc_unews[idx2d];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Kernel for updating halos via P2P
////////////////////////////////////////////////////////////////////////////////
__global__ void update_halo(_DOUBLE_ * __restrict u_target, _DOUBLE_ * __restrict u_source, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int pitch, const unsigned int p)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int k_off = pitch*(Ny+2);

	unsigned int ks = _Nz + (1-p)*(1-_Nz); //source k value
	unsigned int kt = (1-p)*(_Nz+1); //target k value (ghost cell k index)

	unsigned int idx3ds = i + j * pitch + ks * k_off;
	unsigned int idx3dt = i + j * pitch + kt * k_off;

	if( i < Nx+2 && j < Ny+2 && ks < _Nz+2 && kt < _Nz+2 )
	{
		u_target[idx3dt] = u_source[idx3ds];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Kernel for computing the 3D Heat equation async on the GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void heat3D_async(_DOUBLE_ * __restrict u_new, _DOUBLE_ * __restrict u_old, const unsigned int j_off, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int kstart, const unsigned int kstop, const unsigned int loop_z)
{
	register _DOUBLE_ center;
	register _DOUBLE_ north_east;
	register _DOUBLE_ south_west;

    unsigned int i = 1+threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = 1+threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int k = 1+ blockIdx.z * loop_z;

    k = max(kstart,k);

	unsigned int k_off = j_off*(Ny+2);


    unsigned int idx = i + j*j_off + k*k_off;

    if (i > 0 && i < Nx+1 && j > 0 && j < Ny+1)
    {
		south_west = u_old[idx-k_off];
    	center = u_old[idx];
    	north_east = u_old[idx+k_off];
    	u_new[idx] = d_c1 * center + d_c0 * (u_old[idx-1] + u_old[idx+1] + u_old[idx-j_off] + u_old[idx+j_off] + south_west + north_east);

    	for(unsigned int z = 1; z < loop_z; z++)
    	{
    		k += 1;

    		if (k < min(kstop, _Nz+1))
    		{
    			idx = idx+k_off;

				south_west = center;
				center = north_east;
				north_east = u_old[idx+k_off];
				u_new[idx] = (d_c1 * center) + d_c0 * (u_old[idx-1] + u_old[idx+1] + u_old[idx-j_off] + u_old[idx+j_off] + south_west + north_east);
	  		}
    	}
    }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel for computing 3D Heat equation on the GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void heat3D(_DOUBLE_ * __restrict u_new, _DOUBLE_ * __restrict u_old, const unsigned int j_off, const unsigned int Nx, const unsigned int Ny, const unsigned int _Nz, const unsigned int loop_z)
{
	register _DOUBLE_ center;
	register _DOUBLE_ north_east;
	register _DOUBLE_ south_west;

    unsigned int i = 1+threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = 1+threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int k = 1+threadIdx.z + blockIdx.z * loop_z;

	unsigned int k_off = j_off*(Ny+2);

    unsigned int idx = i + j*j_off + k*k_off;

    if (i > 0 && i < Nx+1 && j > 0 && j < Ny+1)
    {
		south_west = u_old[idx-k_off];
    	center = u_old[idx];
    	north_east = u_old[idx+k_off];
    	u_new[idx] = d_c1 * center + d_c0 * (u_old[idx-1] + u_old[idx+1] + u_old[idx-j_off] + u_old[idx+j_off] + south_west + north_east);

    	for(unsigned int z = 1; z < loop_z; z++)
    	{
    		idx = idx+k_off;
    		south_west = center;
    		center = north_east;
    		north_east = u_old[idx+k_off];
    		u_new[idx] = (d_c1 * center) + d_c0 * (u_old[idx-1] + u_old[idx+1] + u_old[idx-j_off] + u_old[idx+j_off] + south_west + north_east);
    	}
    }
}

/////////////////////////////////////////////////////////////////////////////
// Function that copies content from the host to the device's constant memory
/////////////////////////////////////////////////////////////////////////////
extern "C" void CopyToConstantMemory(const _DOUBLE_ c0, const _DOUBLE_ c1)
{
	checkCuda(cudaMemcpyToSymbol(d_c0, &c0, sizeof(_DOUBLE_), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(d_c1, &c1, sizeof(_DOUBLE_), 0, cudaMemcpyHostToDevice));
}

extern "C" void ComputeInnerPoints(dim3 thread_blocks, dim3 threads_per_block, _DOUBLE_* d_s_Unews, _DOUBLE_* d_s_Uolds,
		int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz)
{
	heat3D<<<thread_blocks, threads_per_block>>>(d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, k_loop);
}

extern "C" void ComputeInnerPointsAsync(dim3 thread_blocks, dim3 threads_per_block, cudaStream_t aStream, _DOUBLE_* d_s_Unews, _DOUBLE_* d_s_Uolds,
		int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int kstart, unsigned int kstop)
{
	heat3D_async<<<thread_blocks, threads_per_block, 0, aStream>>>(d_s_Unews, d_s_Uolds, pitch, Nx, Ny, _Nz, kstart, kstop, k_loop);
}

extern "C" void CopyBoundaryRegionToGhostCell(dim3 thread_blocks_halo, dim3 threads_per_block, _DOUBLE_* d_s_Unews, _DOUBLE_* d_right_send_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_br_to_gc<<<thread_blocks_halo, threads_per_block>>>(d_s_Unews, d_right_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, _DOUBLE_* d_s_Unews, _DOUBLE_* d_right_send_buffer,
		unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_br_to_gc<<<thread_blocks_halo, threads_per_block, 0, aStream>>>(d_s_Unews, d_right_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyGhostCellToBoundaryRegion(dim3 thread_blocks_halo, dim3 threads_per_block, _DOUBLE_* d_s_Unews, _DOUBLE_* d_left_receive_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_gc_to_br<<<thread_blocks_halo, threads_per_block>>>(d_s_Unews, d_left_receive_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, _DOUBLE_* d_s_Unews, _DOUBLE_* d_receive_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, int pitch, int gc_pitch, int p)
{
	copy_gc_to_br<<<thread_blocks_halo, threads_per_block, 0, aStream>>>(d_s_Unews, d_receive_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}
