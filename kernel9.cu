#include "common.h"

#define BLOCK_DIM 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_DIM / WARP_SIZE)
#define ROW_TILE_SIZE 4

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
    extern __shared__ float sharedMem[];
    float* row = sharedMem;

    __shared__ unsigned int nnz;
    __shared__ float warpReduction[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned int outputIdx;

    const unsigned int tid = threadIdx.x;
    const unsigned int globalRowOffset = blockIdx.x * ROW_TILE_SIZE;

    if (tid == 0) nnz = 0;

    for (unsigned int i = tid; i < numCols2; i += blockDim.x) row[i] = 0.0f;
    __syncthreads();

    for (unsigned int rowOffset = 0; rowOffset < ROW_TILE_SIZE; rowOffset++) {
        const unsigned int rowA = globalRowOffset + rowOffset;
        if (rowA >= numRows1) continue;

        const unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
        const unsigned int rowEnd1   = csrMatrix1->rowPtrs[rowA + 1];
        if (rowStart1 == rowEnd1) continue;

        for (unsigned int i = rowStart1 + tid; i < rowEnd1; i += blockDim.x) {
            const float valA = csrMatrix1->values[i];
            const unsigned int colA = csrMatrix1->colIdxs[i];
            const unsigned int rowB = colA;
            if (rowB >= numRows2) continue;

            const unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
            const unsigned int rowEnd2   = csrMatrix2->rowPtrs[rowB + 1];

            for (unsigned int j = rowStart2; j < rowEnd2; j++) {
                const unsigned int colB = csrMatrix2->colIdxs[j];
                const float valB = csrMatrix2->values[j];
                const float product = valA * valB;
                if (product != 0.0f) {
                    float oldVal = atomicAdd(&row[colB], product);
                    if (oldVal == 0.0f) atomicAdd(&nnz, 1);
                }
            }
        }

        __syncthreads();

        if (nnz > 0) {
            if (tid == 0) outputIdx = atomicAdd(&cooMatrix3->numNonzeros, nnz);
            __syncthreads();
            for (unsigned int i = tid; i < numCols2; i += blockDim.x) {
                const float val = row[i];
                if (val != 0.0f) {
                    const unsigned int pos = atomicAdd(&outputIdx, 1);
                    cooMatrix3->rowIdxs[pos] = rowA;
                    cooMatrix3->colIdxs[pos] = i;
                    cooMatrix3->values[pos] = val;
                    row[i] = 0.0f;
                }
            }
            if (tid == 0) nnz = 0;
        }
        __syncthreads();
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
    const dim3 grid((numRows1 + ROW_TILE_SIZE - 1) / ROW_TILE_SIZE);
    const size_t sharedMemSize = numCols2 * sizeof(float);

    spmspm_kernel9<<<grid, block, sharedMemSize>>>(
        cooMatrix1,
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
