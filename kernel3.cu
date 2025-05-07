#include "common.h"

#define BLOCK_DIM 32

__global__ void spmspm_kernel3(COOMatrix *cooMatrix1,
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
    __shared__ float row[BLOCK_DIM];

    unsigned int rowA = blockIdx.x;
    unsigned int colIdx = threadIdx.x;
    unsigned int globalColIdx = blockIdx.y * BLOCK_DIM + threadIdx.x;

    if (globalColIdx < numCols2 && colIdx < BLOCK_DIM)
    {
        row[colIdx] = 0.0f;
        __syncthreads();

        unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
        unsigned int rowEnd1 = csrMatrix1->rowPtrs[rowA + 1];

        for (unsigned int i = rowStart1 + threadIdx.y; i < rowEnd1; i += blockDim.y)
        {
            float valA = csrMatrix1->values[i];
            unsigned int rowB = csrMatrix1->colIdxs[i];

            unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
            unsigned int rowEnd2 = csrMatrix2->rowPtrs[rowB + 1];

            for (unsigned int j = rowStart2; j < rowEnd2; ++j)
            {
                unsigned int colB = csrMatrix2->colIdxs[j];
                float valB = csrMatrix2->values[j];
                if (colB == globalColIdx)
                {
                    atomicAdd(&row[colIdx], valA * valB);
                }
            }
        }

        __syncthreads();

        if (threadIdx.y == 0 && row[colIdx] != 0.0f)
        {
            unsigned int numNonzeros = atomicAdd(&cooMatrix3->numNonzeros, 1);
            cooMatrix3->rowIdxs[numNonzeros] = rowA;
            cooMatrix3->colIdxs[numNonzeros] = globalColIdx;
            cooMatrix3->values[numNonzeros] = row[colIdx];
        }
    }
}

void spmspm_gpu3(COOMatrix *cooMatrix1,
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
    const dim3 block(BLOCK_DIM, 4);
    const dim3 grid(numRows1, (numCols2 + block.x - 1) / block.x);

    spmspm_kernel3<<<grid, block>>>(cooMatrix1,
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