#include "common.h"

#define BLOCK_DIM 64

__global__ void spmspm_kernel8(COOMatrix *cooMatrix1,
                               CSRMatrix *csrMatrix1,
                               CSCMatrix *cscMatrix1,
                               COOMatrix *cooMatrix2,
                               CSRMatrix *csrMatrix2,
                               CSCMatrix *cscMatrix2,
                               COOMatrix *cooMatrix3,
                               const unsigned int numRows1,
                               const unsigned int numRows2,
                               const unsigned int numCols2,
                               const unsigned int numNonzeros1,
                               const unsigned int numNonzeros2)
{
    extern __shared__ float row[];
    __shared__ unsigned int nnz;

    if (threadIdx.x == 0) nnz = 0;

    for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x) row[i] = 0.0f;
    __syncthreads();

    const unsigned int rowA      = blockIdx.x;
    const unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
    const unsigned int rowEnd1   = csrMatrix1->rowPtrs[rowA + 1];

    for (unsigned int i = rowStart1 + threadIdx.x; i < rowEnd1; i += blockDim.x)
    {
        const float valA        = csrMatrix1->values[i];
        const unsigned int rowB = csrMatrix1->colIdxs[i];

        const unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
        const unsigned int rowEnd2   = csrMatrix2->rowPtrs[rowB + 1];

        #pragma unroll
        for (unsigned int j = rowStart2; j < rowEnd2; ++j)
        {
            const unsigned int colB = csrMatrix2->colIdxs[j];
            const float valB        = csrMatrix2->values[j];
            const float prod        = valA * valB;

            if (prod != 0.0f)
            {
                float old = atomicAdd(&row[colB], prod);
                if (old == 0.0f) atomicAdd(&nnz, 1);
            }
        }
    }
    __syncthreads();

    if (nnz != 0)
    {
        __shared__ unsigned int baseIdx;
        if (threadIdx.x == 0) baseIdx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
        __syncthreads();

        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        {
            float v = row[i];
            if (v != 0.0f)
            {
                unsigned int pos = atomicAdd(&baseIdx, 1);
                cooMatrix3->rowIdxs[pos] = rowA;
                cooMatrix3->colIdxs[pos] = i;
                cooMatrix3->values [pos] = v;
            }
        }
    }
}

void spmspm_gpu8(COOMatrix *cooMatrix1,
                 CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1,
                 COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2,
                 CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3,
                 unsigned int numRows1,
                 unsigned int numRows2,
                 unsigned int numCols2,
                 unsigned int numNonzeros1,
                 unsigned int numNonzeros2)
{
    const dim3 block(BLOCK_DIM);
    const dim3 grid(numRows1);
    const size_t shMemBytes = numCols2 * sizeof(float);

    spmspm_kernel8<<<grid, block, shMemBytes>>>(
        cooMatrix1, csrMatrix1, cscMatrix1,
        cooMatrix2, csrMatrix2, cscMatrix2,
        cooMatrix3,
        numRows1, numRows2, numCols2,
        numNonzeros1, numNonzeros2);
}
