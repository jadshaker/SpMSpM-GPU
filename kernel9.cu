#include "common.h"

#define BLOCK_DIM 64

__global__ void spmspm_kernel9(COOMatrix *cooMatrix1,
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

    unsigned int rowA = blockIdx.x;
    for (unsigned int c = threadIdx.x; c < cscMatrix1->numCols; c += blockDim.x)
    {
        unsigned int colStart1 = cscMatrix1->colPtrs[c];
        unsigned int colEnd1   = cscMatrix1->colPtrs[c + 1];
        for (unsigned int idx = colStart1; idx < colEnd1; ++idx)
        {
            if (cscMatrix1->rowIdxs[idx] == rowA)
            {
                float valA = cscMatrix1->values[idx];
                for (unsigned int j = 0; j < cscMatrix2->numCols; j++)
                {
                    unsigned int colStart2 = cscMatrix2->colPtrs[j];
                    unsigned int colEnd2   = cscMatrix2->colPtrs[j + 1];
                    for (unsigned int x2 = colStart2; x2 < colEnd2; x2++)
                    {
                        if (cscMatrix2->rowIdxs[x2] == c)
                        {
                            float valB = cscMatrix2->values[x2];
                            float prod = valA * valB;
                            if (prod != 0.0f)
                            {
                                float oldVal = atomicAdd(&row[j], prod);
                                if (oldVal == 0.0f) atomicAdd(&nnz, 1);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    if (nnz != 0)
    {
        __shared__ unsigned int idx;
        if (threadIdx.x == 0) idx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
        __syncthreads();

        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        {
            float v = row[i];
            if (v != 0.0f)
            {
                unsigned int index = atomicAdd(&idx, 1);
                cooMatrix3->rowIdxs[index] = rowA;
                cooMatrix3->colIdxs[index] = i;
                cooMatrix3->values[index]  = v;
            }
        }
    }
}

void spmspm_gpu9(COOMatrix *cooMatrix1,
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
    size_t shMemBytes = numCols2 * sizeof(float);

    spmspm_kernel9<<<grid, block, shMemBytes>>>(cooMatrix1,
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
