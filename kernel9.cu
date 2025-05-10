#include "common.h"

#define BLOCK_DIM 64            // threads per block
#define TILE_SIZE 64            // how many elements we load at once from csrMatrix2
#define LOCAL_BUFFER_SIZE 8     // increased local buffer size

// Simple warp-level sum for 32-thread warps
__forceinline__ __device__ unsigned int warpReduceSum(unsigned int val)
{
    // Full mask for warp shuffle
    unsigned int mask = 0xffffffff;
    // Warp size is 32 on all current CUDA GPUs
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Flush partial sums in the local buffer to the shared-memory row accumulator
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

__global__ void spmspm_kernel9(COOMatrix *cooMatrix1,
                               CSRMatrix *csrMatrix1,  // first matrix (M x K)
                               CSCMatrix *cscMatrix1,  // not used
                               COOMatrix *cooMatrix2,
                               CSRMatrix *csrMatrix2,  // second matrix (K x N)
                               CSCMatrix *cscMatrix2,  // not used
                               COOMatrix *cooMatrix3,
                               unsigned int numRows1,
                               unsigned int numRows2,
                               unsigned int numCols2,
                               unsigned int numNonzeros1,
                               unsigned int numNonzeros2)
{
    // We will use shared memory for:
    // 1) rowAcc[numCols2] to store partial sums for this row
    // 2) tileCol[TILE_SIZE], tileVal[TILE_SIZE] for loading rowB in chunks
    extern __shared__ char shMem[];
    float      *rowAcc  = reinterpret_cast<float*>(shMem);
    unsigned int *tileCol = reinterpret_cast<unsigned int*>(&rowAcc[numCols2]);
    float      *tileVal = reinterpret_cast<float*>(&tileCol[TILE_SIZE]);

    __shared__ unsigned int nnz;
    if (threadIdx.x == 0) nnz = 0;

    // Initialize row accumulator
    #pragma unroll 4
    for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x) {
        rowAcc[i] = 0.0f;
    }
    __syncthreads();

    // Each block handles one row of the output
    const unsigned int rowA = blockIdx.x;
    const unsigned int rowStartA = csrMatrix1->rowPtrs[rowA];
    const unsigned int rowEndA   = csrMatrix1->rowPtrs[rowA + 1];

    // Per-thread local buffer
    unsigned int localCols[LOCAL_BUFFER_SIZE];
    float        localVals[LOCAL_BUFFER_SIZE];
    int          localCount = 0;

    // Lambda to accumulate partial sums in the local buffer
    auto accumulateLocal = [&](unsigned int c, float v)
    {
        // Check if c is already in buffer
        #pragma unroll
        for (int i = 0; i < localCount; i++)
        {
            if (localCols[i] == c)
            {
                localVals[i] += v;
                return;
            }
        }
        // If not found, see if we have space
        if (localCount < LOCAL_BUFFER_SIZE)
        {
            localCols[localCount] = c;
            localVals[localCount] = v;
            localCount++;
        }
        else
        {
            // Flush buffer if full
            flushBuffer(rowAcc, localCols, localVals, localCount);
            // Then add new entry
            localCols[localCount] = c;
            localVals[localCount] = v;
            localCount++;
        }
    };

    // Process each nonzero in rowA of first matrix
    // for each: (rowA, colB, valA)
    // multiply valA by entire rowB in second matrix
    for (unsigned int i = rowStartA + threadIdx.x; i < rowEndA; i += blockDim.x)
    {
        float valA        = csrMatrix1->values[i];
        unsigned int colB = csrMatrix1->colIdxs[i];

        // rowB in second matrix = rowPtrs[colB]..rowPtrs[colB+1]
        unsigned int rowStartB = csrMatrix2->rowPtrs[colB];
        unsigned int rowEndB   = csrMatrix2->rowPtrs[colB + 1];

        // Tile rowB in chunks of TILE_SIZE
        for (unsigned int tileBegin = rowStartB; tileBegin < rowEndB; tileBegin += TILE_SIZE)
        {
            unsigned int tileEnd = (tileBegin + TILE_SIZE < rowEndB)
                                     ? tileBegin + TILE_SIZE 
                                     : rowEndB;

            // Load chunk of rowB into shared memory
            for (unsigned int t = tileBegin + threadIdx.x; t < tileEnd; t += blockDim.x)
            {
                unsigned int idx = t - tileBegin; // local offset in tile
                tileCol[idx] = csrMatrix2->colIdxs[t];
                tileVal[idx] = csrMatrix2->values[t];
            }
            __syncthreads();

            // Multiply valA by each element in the tile
            unsigned int tileSizeNow = tileEnd - tileBegin;

            // Unroll the loop over tile elements
            #pragma unroll 4
            for (unsigned int k = 0; k < TILE_SIZE; k++)
            {
                if (k < tileSizeNow)
                {
                    float valB        = tileVal[k];
                    unsigned int colC = tileCol[k];
                    float prod        = valA * valB;
                    if (prod != 0.0f)
                        accumulateLocal(colC, prod);
                }
            }
            __syncthreads();
        }
    }

    // Flush leftover local buffer
    flushBuffer(rowAcc, localCols, localVals, localCount);
    __syncthreads();

    // Phase 2: Count how many columns are nonzero using warp-level summation
    {
        // Each thread counts how many nonzero columns it sees
        unsigned int localCountCols = 0;
        #pragma unroll 4
        for (unsigned int i = threadIdx.x; i < numCols2; i += blockDim.x)
        {
            if (rowAcc[i] != 0.0f) localCountCols++;
        }

        // Now do warp-level reduction on localCountCols
        unsigned int warpSum = warpReduceSum(localCountCols);

        // The lane 0 in each warp will add to nnz
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&nnz, warpSum);
        }
    }
    __syncthreads();

    // If nnz > 0, allocate space in cooMatrix3 and write those columns
    if (nnz != 0)
    {
        __shared__ unsigned int baseIdx;
        if (threadIdx.x == 0) {
            baseIdx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
        }
        __syncthreads();

        // Now we each find and write the columns we own
        // (We won't do warp-level merges here, but could if we wanted.)
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
    // Shared memory usage:
    //   rowAcc[numCols2] floats
    // + tileCol[TILE_SIZE] uint
    // + tileVal[TILE_SIZE] float
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
