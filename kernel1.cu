#include "common.h"
#include <stdio.h>

#define TILE_SIZE 32

__global__ void spmspm_kernel1(const CSRMatrix* csrA,
                               const CSCMatrix* cscB,
                               COOMatrix*       cooC,
                               unsigned int*    d_nnz,
                               unsigned int     numRowsA,
                               unsigned int     numColsB)
{
    __shared__ unsigned int s_rowPtrs[TILE_SIZE + 1];

    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int rowBlockStart = blockIdx.x * blockDim.x;
    if (tid < blockDim.x + 1 && rowBlockStart + tid <= numRowsA) {
        s_rowPtrs[tid] = csrA->rowPtrs[rowBlockStart + tid];
    }
    __syncthreads();

    if (row >= numRowsA || col >= numColsB) return;

    unsigned int rowStart, rowEnd;
    if (threadIdx.x < blockDim.x && row + 1 <= numRowsA) {
        rowStart = s_rowPtrs[threadIdx.x];
        rowEnd   = s_rowPtrs[threadIdx.x + 1];
    } else {
        rowStart = csrA->rowPtrs[row];
        rowEnd   = csrA->rowPtrs[row + 1];
    }

    unsigned int colStart = cscB->colPtrs[col];
    unsigned int colEnd   = cscB->colPtrs[col + 1];

    float sum = 0.0f;

    for (unsigned int r = rowStart; r < rowEnd; ++r) {
        unsigned int aCol = csrA->colIdxs[r];
        float        aVal = csrA->values[r];

        for (unsigned int c = colStart; c < colEnd; ++c) {
            if (aCol == cscB->rowIdxs[c]) {
                sum += aVal * cscB->values[c];
            }
        }
    }

    if (sum != 0.0f) {
        unsigned int pos = atomicAdd(d_nnz, 1);
        if (pos < cooC->capacity) {
            cooC->rowIdxs[pos] = row;
            cooC->colIdxs[pos] = col;
            cooC->values[pos]  = sum;
        }
    }
}

void spmspm_gpu1(COOMatrix*,
                 CSRMatrix*  csrMatrix1_d,
                 CSCMatrix*,
                 COOMatrix*,
                 CSRMatrix*,
                 CSCMatrix*  cscMatrix2_d,
                 COOMatrix*  cooMatrix3_d,
                 unsigned int numRows1,
                 unsigned int,
                 unsigned int numCols2,
                 unsigned int,
                 unsigned int)
{
    unsigned int* d_nnz;
    unsigned int  zero = 0;
    CUDA_ERROR_CHECK(cudaMalloc(&d_nnz, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_nnz, &zero, sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((numRows1 + block.x - 1) / block.x,
                    (numCols2 + block.y - 1) / block.y);

    spmspm_kernel1<<<grid, block>>>(csrMatrix1_d,
                                    cscMatrix2_d,
                                    cooMatrix3_d,
                                    d_nnz,
                                    numRows1,
                                    numCols2);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CUDA_ERROR_CHECK(cudaMemcpy(&(cooMatrix3_d->numNonzeros), d_nnz,
                                sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaFree(d_nnz);
}
