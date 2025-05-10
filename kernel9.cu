#include "common.h"

#define BLOCK_DIM 64

__global__ void spmspm_kernel9(COOMatrix *cooMatrix1,
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
    extern __shared__ float rowAcc[];
    __shared__ unsigned int nnz;
    if (threadIdx.x == 0) nnz = 0;
    for (unsigned int i = threadIdx.x; i < cscMatrix1->numRows; i += blockDim.x) rowAcc[i] = 0.0f;
    __syncthreads();
    unsigned int colB = blockIdx.x;
    if (colB < cscMatrix2->numCols)
    {
        unsigned int startB = cscMatrix2->colPtrs[colB];
        unsigned int endB   = cscMatrix2->colPtrs[colB + 1];
        for (unsigned int i = startB + threadIdx.x; i < endB; i += blockDim.x)
        {
            float valB         = cscMatrix2->values[i];
            unsigned int colA  = cscMatrix2->rowIdxs[i];
            unsigned int startA = cscMatrix1->colPtrs[colA];
            unsigned int endA   = cscMatrix1->colPtrs[colA + 1];
            for (unsigned int j = startA; j < endA; j++)
            {
                float valA = cscMatrix1->values[j];
                unsigned int rA = cscMatrix1->rowIdxs[j];
                float prod = valA * valB;
                if (prod != 0.0f) atomicAdd(&rowAcc[rA], prod);
            }
        }
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < cscMatrix1->numRows; i += blockDim.x)
            if (rowAcc[i] != 0.0f) atomicAdd(&nnz, 1);
        __syncthreads();
        if (nnz != 0)
        {
            __shared__ unsigned int baseIdx;
            if (threadIdx.x == 0) baseIdx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < cscMatrix1->numRows; i += blockDim.x)
            {
                float v = rowAcc[i];
                if (v != 0.0f)
                {
                    unsigned int pos = atomicAdd(&baseIdx, 1);
                    cooMatrix3->rowIdxs[pos] = i;
                    cooMatrix3->colIdxs[pos] = colB;
                    cooMatrix3->values[pos]  = v;
                }
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
    dim3 block(BLOCK_DIM);
    dim3 grid(cscMatrix2->numCols);
    size_t shMemBytes = cscMatrix1->numRows * sizeof(float);
    spmspm_kernel9<<<grid, block, shMemBytes>>>(cooMatrix1, csrMatrix1, cscMatrix1,
                                                cooMatrix2, csrMatrix2, cscMatrix2,
                                                cooMatrix3,
                                                numRows1, numRows2, numCols2,
                                                numNonzeros1, numNonzeros2);
}
