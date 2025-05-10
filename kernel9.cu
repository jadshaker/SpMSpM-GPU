#include "common.h"

#define BLOCK_DIM 64
#define TILE_SIZE 64
#define LOCAL_BUFFER_SIZE 8

__forceinline__ __device__
unsigned int warpReduceSum(unsigned int val)
{
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__
void flushBuffer(float       *rowAcc, 
                 unsigned int *bufCols,
                 float       *bufVals,
                 int         &bufCount)
{
    #pragma unroll
    for (int i = 0; i < bufCount; i++)
    {
        atomicAdd(&rowAcc[bufCols[i]], bufVals[i]);
        bufVals[i] = 0.0f;
    }
    bufCount = 0;
}

__device__ __forceinline__
void accumulateLocal(unsigned int col, float val,
                     float *rowAcc,
                     unsigned int *bufCols,
                     float *bufVals,
                     int &bufCount)
{
    #pragma unroll
    for (int i = 0; i < bufCount; i++)
    {
        if (bufCols[i] == col)
        {
            bufVals[i] += val;
            return;
        }
    }
    if (bufCount < LOCAL_BUFFER_SIZE)
    {
        bufCols[bufCount] = col;
        bufVals[bufCount] = val;
        bufCount++;
    }
    else
    {
        flushBuffer(rowAcc, bufCols, bufVals, bufCount);
        // Now insert the new entry
        bufCols[bufCount] = col;
        bufVals[bufCount] = val;
        bufCount++;
    }
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
    extern __shared__ char shMem[];
    float       *rowAcc  = reinterpret_cast<float*>(shMem);
    unsigned int *tileCol = reinterpret_cast<unsigned int*>(&rowAcc[numCols2]);
    float       *tileVal = reinterpret_cast<float*>(&tileCol[TILE_SIZE]);

    __shared__ unsigned int nnz;
    if (threadIdx.x == 0) nnz = 0;

    #pragma unroll 4
    for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        rowAcc[i] = 0.0f;
    __syncthreads();

    const unsigned int rowA      = blockIdx.x;
    const unsigned int rowStartA = csrMatrix1->rowPtrs[rowA];
    const unsigned int rowEndA   = csrMatrix1->rowPtrs[rowA + 1];

    unsigned int localCols[LOCAL_BUFFER_SIZE];
    float        localVals[LOCAL_BUFFER_SIZE];
    int          localCount = 0;

    for (unsigned int i = rowStartA + threadIdx.x; i < rowEndA; i += blockDim.x)
    {
        float valA        = csrMatrix1->values[i];
        unsigned int colB = csrMatrix1->colIdxs[i];

        unsigned int rowStartB = csrMatrix2->rowPtrs[colB];
        unsigned int rowEndB   = csrMatrix2->rowPtrs[colB + 1];

        for (unsigned int tileBegin = rowStartB; tileBegin < rowEndB; tileBegin += TILE_SIZE)
        {
            unsigned int tileEnd = (tileBegin + TILE_SIZE < rowEndB)
                                    ? tileBegin + TILE_SIZE 
                                    : rowEndB;

            for (unsigned int t = tileBegin + threadIdx.x; t < tileEnd; t += blockDim.x)
            {
                unsigned int idx = t - tileBegin;
                tileCol[idx] = csrMatrix2->colIdxs[t];
                tileVal[idx] = csrMatrix2->values[t];
            }
            __syncthreads();

            unsigned int tileSizeNow = tileEnd - tileBegin;

            #pragma unroll 4
            for (unsigned int k = 0; k < TILE_SIZE; k++)
            {
                if (k < tileSizeNow)
                {
                    float valB        = tileVal[k];
                    unsigned int colC = tileCol[k];
                    float prod        = valA * valB;
                    if (prod != 0.0f)
                    {
                        accumulateLocal(colC, prod,
                                        rowAcc,
                                        localCols,
                                        localVals,
                                        localCount);
                    }
                }
            }
            __syncthreads();
        }
    }

    flushBuffer(rowAcc, localCols, localVals, localCount);
    __syncthreads();

    {
        unsigned int localCountCols = 0;
        #pragma unroll 4
        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        {
            if (rowAcc[i] != 0.0f) localCountCols++;
        }

        unsigned int warpSum = warpReduceSum(localCountCols);

        if ((threadIdx.x & 31) == 0)
            atomicAdd(&nnz, warpSum);
    }
    __syncthreads();

    if (nnz > 0)
    {
        __shared__ unsigned int baseIdx;
        if (threadIdx.x == 0)
        {
            baseIdx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
        }
        __syncthreads();

        #pragma unroll 4
        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        {
            float v = rowAcc[i];
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
    size_t shMemBytes = numCols2 * sizeof(float)
                      + TILE_SIZE * sizeof(unsigned int)
                      + TILE_SIZE * sizeof(float);

    dim3 block(BLOCK_DIM);
    dim3 grid(numRows1);

    spmspm_kernel9<<<grid, block, shMemBytes>>>(
        cooMatrix1, csrMatrix1, cscMatrix1,
        cooMatrix2, csrMatrix2, cscMatrix2,
        cooMatrix3,
        numRows1, numRows2, numCols2,
        numNonzeros1, numNonzeros2
    );
}
