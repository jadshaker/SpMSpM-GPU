#include "common.h"

#define BLOCK_DIM 64
#define DS_DIM 5000

__global__ void spmspm_kernel6(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1, CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2, CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2, COOMatrix *cooMatrix3, const unsigned int numRows1, const unsigned int numRows2, const unsigned int numCols2, const unsigned int numNonzeros1, const unsigned int numNonzeros2)
{
    __shared__ float col[DS_DIM];

    for (unsigned int i = threadIdx.x; i < DS_DIM; i += blockDim.x)
    {
        col[i] = 0.0f;
    }

    __syncthreads();

    unsigned int colB = blockIdx.x;

    unsigned int colStart2 = cscMatrix2->colPtrs[colB];
    unsigned int colEnd2 = cscMatrix2->colPtrs[colB + 1];

    for (unsigned int idx = 0; idx < numRows1; idx += DS_DIM)
    {
        for (unsigned int i = threadIdx.x + colStart2; i < colEnd2; i += blockDim.x)
        {
            float valB = cscMatrix2->values[i];
            unsigned int rowB = cscMatrix2->rowIdxs[i];

            unsigned int colA = rowB;
            unsigned int colStart1 = cscMatrix1->colPtrs[colA];
            unsigned int colEnd1 = cscMatrix1->colPtrs[colA + 1];

            for (unsigned int j = colStart1; j < colEnd1; ++j)
            {
                unsigned int rowA = cscMatrix1->rowIdxs[j];

                if (rowA >= idx && rowA < idx + DS_DIM)
                {
                    float valA = cscMatrix1->values[j];
                    float val = valA * valB;

                    atomicAdd(&col[rowA % DS_DIM], val);
                }
            }
        }
        __syncthreads();

        for (unsigned int i = threadIdx.x; i < DS_DIM; i += blockDim.x)
        {
            if (col[i] != 0.0f)
            {
                unsigned int numNonzeros = atomicAdd(&cooMatrix3->numNonzeros, 1);
                cooMatrix3->rowIdxs[numNonzeros] = idx + i;
                cooMatrix3->colIdxs[numNonzeros] = colB;
                cooMatrix3->values[numNonzeros] = col[i];
                col[i] = 0.0f;
            }
        }
        __syncthreads();
    }
}

void spmspm_gpu6(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1, CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2, CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2, COOMatrix *cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2)
{
    const dim3 block(BLOCK_DIM);
    const dim3 grid(numCols2);

    spmspm_kernel6<<<grid, block>>>(cooMatrix1, csrMatrix1, cscMatrix1, cooMatrix2, csrMatrix2, cscMatrix2, cooMatrix3, numRows1, numRows2, numCols2, numNonzeros1, numNonzeros2);
}