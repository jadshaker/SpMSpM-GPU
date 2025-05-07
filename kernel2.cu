#include "common.h"

#define BLOCK_DIM 32

__global__ void spmspm_kernel2(COOMatrix *cooMatrix1,
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

    unsigned int rowA = blockIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x;
    unsigned int colIdx = colStart + threadIdx.x;

    if (threadIdx.x < blockDim.x)
    {
        row[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (rowA < numRows1 && colIdx < numCols2)
    {
        unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
        unsigned int rowEnd1 = csrMatrix1->rowPtrs[rowA + 1];

        for (unsigned int i = rowStart1; i < rowEnd1; ++i)
        {
            float valA = csrMatrix1->values[i];
            unsigned int rowB = csrMatrix1->colIdxs[i];

            unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
            unsigned int rowEnd2 = csrMatrix2->rowPtrs[rowB + 1];

            for (unsigned int j = rowStart2; j < rowEnd2; ++j)
            {
                unsigned int colB = csrMatrix2->colIdxs[j];
                float valB = csrMatrix2->values[j];
                if (colB == colIdx)
                {
                    row[threadIdx.x] += valA * valB;
                }
            }
        }

        __syncthreads();

        if (row[threadIdx.x] != 0.0f)
        {
            unsigned int numNonzeros = atomicAdd(&cooMatrix3->numNonzeros, 1);
            cooMatrix3->rowIdxs[numNonzeros] = rowA;
            cooMatrix3->colIdxs[numNonzeros] = colIdx;
            cooMatrix3->values[numNonzeros] = row[threadIdx.x];
        }
    }
}

void spmspm_gpu2(COOMatrix *cooMatrix1,
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
    const dim3 grid((numCols2 + block.x - 1) / block.x, numRows1);

    spmspm_kernel2<<<grid, block>>>(cooMatrix1,
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