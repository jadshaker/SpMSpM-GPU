#include <stdio.h>
#include "common.h"

__global__
void spmspm_gpu0_kernel(const unsigned int* __restrict__ A_rowPtrs,
                        const unsigned int* __restrict__ A_colIdxs,
                        const float*       __restrict__ A_vals,
                        unsigned int                       A_numRows,
                        const unsigned int* __restrict__ B_colPtrs,
                        const unsigned int* __restrict__ B_rowIdxs,
                        const float*       __restrict__ B_vals,
                        unsigned int                       B_numCols,
                        unsigned int*      __restrict__ C_rowIdxs,
                        unsigned int*      __restrict__ C_colIdxs,
                        float*            __restrict__  C_vals,
                        unsigned int*      __restrict__ d_count,
                        unsigned int                       capacity)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if(row>=A_numRows || col>=B_numCols) return;

    unsigned int rStart = A_rowPtrs[row];
    unsigned int rEnd   = A_rowPtrs[row+1];
    unsigned int cStart = B_colPtrs[col];
    unsigned int cEnd   = B_colPtrs[col+1];

    float val=0.0f;

    for(unsigned int i=rStart; i<rEnd; i++){
        unsigned int aCol = A_colIdxs[i];
        float aVal = A_vals[i];
        for(unsigned int j=cStart; j<cEnd; j++){
            if(aCol == B_rowIdxs[j]){
                val += aVal * B_vals[j];
            }
        }
    }
    if(val!=0.f){
        unsigned int pos = atomicAdd(d_count,1);
        if(pos>=capacity){
            // avoid out‑of‑bounds
            atomicSub(d_count,1);
            return;
        }
        C_rowIdxs[pos] = row;
        C_colIdxs[pos] = col;
        C_vals[pos]    = val;
    }
}

void spmspm_gpu0(COOMatrix* /*A_coo*/,
                 CSRMatrix*  A_csr,
                 CSCMatrix*  /*A_csc*/,
                 COOMatrix* /*B_coo*/,
                 CSRMatrix*  /*B_csr*/,
                 CSCMatrix*  B_csc,
                 COOMatrix*  C_coo,
                 unsigned int /*numRows1*/,
                 unsigned int /*numRows2*/,
                 unsigned int /*numCols2*/,
                 unsigned int /*numNonzeros1*/,
                 unsigned int /*numNonzeros2*/)
{
    // We'll do 2D grid: (row, col)
    unsigned int Arows = A_csr->numRows;
    unsigned int Bcols = B_csc->numCols;

    unsigned int *d_count=nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_count,sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemset(d_count,0,sizeof(unsigned int)));

    dim3 block(16,16);
    dim3 grid((Arows+block.x-1)/block.x,(Bcols+block.y-1)/block.y);

    spmspm_gpu0_kernel<<<grid, block>>>(
        A_csr->rowPtrs, A_csr->colIdxs, A_csr->values, A_csr->numRows,
        B_csc->colPtrs, B_csc->rowIdxs, B_csc->values, B_csc->numCols,
        C_coo->rowIdxs, C_coo->colIdxs, C_coo->values,
        d_count, C_coo->capacity);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    unsigned int h_count=0;
    CUDA_ERROR_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_count));

    // write final counts
    // must copy struct from device, update numNonzeros, then copy back
    COOMatrix hostStruct;
    CUDA_ERROR_CHECK(cudaMemcpy(&hostStruct, C_coo, sizeof(COOMatrix), cudaMemcpyDeviceToHost));

    hostStruct.numNonzeros = h_count;
    hostStruct.numRows     = Arows;
    hostStruct.numCols     = Bcols;

    CUDA_ERROR_CHECK(cudaMemcpy(C_coo, &hostStruct, sizeof(COOMatrix), cudaMemcpyHostToDevice));
}
