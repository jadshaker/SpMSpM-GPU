#include "common.h"

#define BLOCK_DIM 64

__global__ void spmspm_kernel5(COOMatrix *cooMatrix1,
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

    for (int i = threadIdx.x; i < numCols2; i += blockDim.x)
        row[i] = 0;
    __syncthreads();

    unsigned int rowA = blockIdx.x;
    unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
    unsigned int rowEnd1 = csrMatrix1->rowPtrs[rowA + 1];

    for (unsigned int i = rowStart1 + threadIdx.x; i < rowEnd1; i += blockDim.x)
    {
        float valA = csrMatrix1->values[i];
        unsigned int rowB = csrMatrix1->colIdxs[i];

        unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
        unsigned int rowEnd2 = csrMatrix2->rowPtrs[rowB + 1];

        for (unsigned int j = rowStart2; j < rowEnd2; ++j)
        {
            unsigned int colB = csrMatrix2->colIdxs[j];
            float valB = csrMatrix2->values[j];
            atomicAdd(&row[colB], valA * valB);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numCols2; i += blockDim.x)
    {
        float v = row[i];
        if (v != 0.0f)
        {
            unsigned int idx = atomicAdd(&cooMatrix3->numNonzeros, 1);
            cooMatrix3->rowIdxs[idx] = rowA;
            cooMatrix3->colIdxs[idx] = i;
            cooMatrix3->values[idx] = v;
        }
    }
}

void spmspm_gpu5(COOMatrix *cooMatrix1,
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

    spmspm_kernel5<<<grid, block, numCols2 * sizeof(float)>>>(cooMatrix1,
                                                              csrMatrix1,
                                                              cscMatrix1,
                                                              cooMatrix2,
                                                              csrMatrix2,
                                                              cscMatrix2,
                                                              cooMatrix3,
                                                              numRows1,
                                                              numRows2,
                                                              numCols2,
                                                              numNonzeros1,
                                                              numNonzeros2);
}