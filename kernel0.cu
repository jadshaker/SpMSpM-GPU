// kernel0.cu
#include "common.h"
#include <stdio.h>   // needed for possible debug prints

/*---------------------------------------------------------------------------
 *  Kernel: one thread does the entire SpM×SpM work (naive, no atomics)
 *---------------------------------------------------------------------------*/
__global__ void spmspm_kernel0(const CSRMatrix* csrMat1,
                               const CSCMatrix* cscMat2,
                               COOMatrix*       cooMat3,
                               unsigned int     numRows1,
                               unsigned int     numCols2)
{
    /* Only thread (0,0) performs the computation */
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int nnz = 0;   // local non‑zero counter

    /* Outer two loops: rows of A (CSR) × columns of B (CSC) */
    for (unsigned int i = 0; i < numRows1; ++i)
    {
        unsigned int rowStart = csrMat1->rowPtrs[i];
        unsigned int rowEnd   = csrMat1->rowPtrs[i + 1];

        for (unsigned int j = 0; j < numCols2; ++j)
        {
            unsigned int colStart = cscMat2->colPtrs[j];
            unsigned int colEnd   = cscMat2->colPtrs[j + 1];

            float sum = 0.0f;

            /* Match column index of A with row index of B */
            for (unsigned int rIdx = rowStart; rIdx < rowEnd; ++rIdx)
            {
                unsigned int aCol = csrMat1->colIdxs[rIdx];
                float        aVal = csrMat1->values[rIdx];

                for (unsigned int cIdx = colStart; cIdx < colEnd; ++cIdx)
                {
                    if (aCol == cscMat2->rowIdxs[cIdx])
                    {
                        sum += aVal * cscMat2->values[cIdx];
                    }
                }
            }

            if (sum != 0.0f)
            {
                /* Write the triplet into the COO result */
                cooMat3->rowIdxs[nnz] = i;
                cooMat3->colIdxs[nnz] = j;
                cooMat3->values[nnz]  = sum;
                ++nnz;
            }
        }
    }

    /* Store total number of non‑zeros */
    cooMat3->numNonzeros = nnz;
}

/*---------------------------------------------------------------------------
 *  Host wrapper: launches the single‑thread kernel
 *---------------------------------------------------------------------------*/
void spmspm_gpu0(COOMatrix* cooMatrix1,
                 CSRMatrix* csrMatrix1,
                 CSCMatrix* cscMatrix1,
                 COOMatrix* cooMatrix2,
                 CSRMatrix* csrMatrix2,
                 CSCMatrix* cscMatrix2,
                 COOMatrix* cooMatrix3,
                 unsigned int numRows1,
                 unsigned int /*numRows2*/,
                 unsigned int numCols2,
                 unsigned int /*numNonzeros1*/,
                 unsigned int /*numNonzeros2*/)
{
    /* One block, one thread – no parallelism, just runs on the GPU */
    dim3 grid(1);
    dim3 block(1);

    spmspm_kernel0<<<grid, block>>>(csrMatrix1,       // A in CSR  (device)
                                    cscMatrix2,       // B in CSC  (device)
                                    cooMatrix3,       // C in COO  (device)
                                    numRows1,
                                    numCols2);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
