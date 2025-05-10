#include "common.h"

#define BLOCK_DIM 64
#define LOCAL_BUFFER_SIZE 4

__device__ __forceinline__
void flushBuffer(float       *rowAcc, 
                 unsigned int *bufCols,
                 float       *bufVals,
                 int         &bufCount)
{
    for (int i = 0; i < bufCount; i++)
    {
        atomicAdd(&rowAcc[bufCols[i]], bufVals[i]);
        bufVals[i] = 0.0f;
    }
    bufCount = 0;
}

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
    extern __shared__ float row[];
    __shared__ unsigned int nnz;

    if (threadIdx.x == 0) nnz = 0;

    // Initialize shared memory accumulator
    for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        row[i] = 0.0f;
    __syncthreads();

    unsigned int rowA = blockIdx.x;
    unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
    unsigned int rowEnd1   = csrMatrix1->rowPtrs[rowA + 1];

    // Thread‐local buffer
    unsigned int localCols[LOCAL_BUFFER_SIZE];
    float        localVals[LOCAL_BUFFER_SIZE];
    int          localCount = 0;

    auto accumulateLocal = [&](unsigned int c, float v) {
        // Try to merge with existing slot
        for (int i = 0; i < localCount; i++)
        {
            if (localCols[i] == c)
            {
                localVals[i] += v;
                return;
            }
        }
        // If no merge, check if buffer has space
        if (localCount < LOCAL_BUFFER_SIZE)
        {
            localCols[localCount] = c;
            localVals[localCount] = v;
            localCount++;
        }
        else
        {
            // Flush buffer to shared memory
            flushBuffer(row, localCols, localVals, localCount);
            // Add new entry
            localCols[localCount] = c;
            localVals[localCount] = v;
            localCount++;
        }
    };

    // Accumulate partial products
    for (unsigned int i = rowStart1 + threadIdx.x; i < rowEnd1; i += blockDim.x)
    {
        float valA = csrMatrix1->values[i];
        unsigned int rowB = csrMatrix1->colIdxs[i];

        unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
        unsigned int rowEnd2   = csrMatrix2->rowPtrs[rowB + 1];

        for (unsigned int j = rowStart2; j < rowEnd2; ++j)
        {
            unsigned int colB = csrMatrix2->colIdxs[j];
            float valB        = csrMatrix2->values[j];
            float prod        = valA * valB;

            if (prod != 0.0f)
                accumulateLocal(colB, prod);
        }
    }
    // Flush any leftover items
    flushBuffer(row, localCols, localVals, localCount);
    __syncthreads();

    // Count how many columns are nonzero
    {
        unsigned int localCountCols = 0;
        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
            if (row[i] != 0.0f)
                localCountCols++;
        atomicAdd(&nnz, localCountCols);
    }
    __syncthreads();

    // Write to cooMatrix3
    if (nnz != 0)
    {
        __shared__ unsigned int baseIdx;
        if (threadIdx.x == 0)
            baseIdx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
        __syncthreads();

        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        {
            float v = row[i];
            if (v != 0.0f)
            {
                unsigned int pos = atomicAdd(&baseIdx, 1);
                cooMatrix3->rowIdxs[pos] = rowA;
                cooMatrix3->colIdxs[pos] = i;
                cooMatrix3->values[pos]  = v;
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
    // One block per row, as with your existing approach
    dim3 block(BLOCK_DIM);
    dim3 grid(numRows1);

    size_t shMemBytes = numCols2 * sizeof(float);

    spmspm_kernel9<<<grid, block, shMemBytes>>>(
        cooMatrix1, csrMatrix1, cscMatrix1,
        cooMatrix2, csrMatrix2, cscMatrix2,
        cooMatrix3,
        numRows1, numRows2, numCols2,
        numNonzeros1, numNonzeros2
    );
}
