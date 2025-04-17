#include "common.h"
#include <stdio.h>

__global__ void spmspm_kernel0(const CSRMatrix* csrA,
                               const CSCMatrix* cscB,
                               COOMatrix*       cooC,
                               unsigned int*    d_nnz,
                               unsigned int     numRowsA,
                               unsigned int     numColsB)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= numRowsA || j >= numColsB) return;

    unsigned int rowStart = csrA->rowPtrs[i];
    unsigned int rowEnd   = csrA->rowPtrs[i + 1];

    unsigned int colStart = cscB->colPtrs[j];
    unsigned int colEnd   = cscB->colPtrs[j + 1];

    float sum = 0.f;

    for (unsigned int r = rowStart; r < rowEnd; ++r) {
        unsigned int aCol = csrA->colIdxs[r];
        float        aVal = csrA->values[r];

        for (unsigned int c = colStart; c < colEnd; ++c) {
            if (aCol == cscB->rowIdxs[c])
                sum += aVal * cscB->values[c];
        }
    }

    if (sum != 0.f) {
        unsigned int pos = atomicAdd(d_nnz, 1u);
        if (pos < cooC->capacity) {
            cooC->rowIdxs[pos] = i;
            cooC->colIdxs[pos] = j;
            cooC->values[pos]  = sum;
        }
    }
}

void spmspm_gpu0(COOMatrix*,
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
    unsigned int zero = 0;
    CUDA_ERROR_CHECK(cudaMalloc(&d_nnz, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_nnz, &zero, sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((numRows1 + block.x - 1) / block.x,
                    (numCols2 + block.y - 1) / block.y);

    spmspm_kernel0<<<grid, block>>>(csrMatrix1_d,
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
