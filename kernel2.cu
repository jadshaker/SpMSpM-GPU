#include "common.h"
#include <stdio.h>
#define TILE_SIZE     16  
#define MAX_BLOCK_NNZ 1024  

typedef struct {
    unsigned int rowIdx;
    unsigned int colIdx;
    float        value;
} COOEntry;

__global__ void spmspm_kernel2(
    const CSRMatrix* csrA,
    const CSCMatrix* cscB,
    COOMatrix*       cooC,
    unsigned int*    d_nnz,
    unsigned int     numRowsA,
    unsigned int     numColsB)
{
    __shared__ unsigned int s_rowPtrs[TILE_SIZE + 1];
    __shared__ COOEntry  s_cooEntries[MAX_BLOCK_NNZ];
    __shared__ unsigned int s_blockNnz;

    __shared__ unsigned int s_stored;
    __shared__ unsigned int s_blockPos;

    const unsigned tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned row  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned col  = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned rowBlockStart = blockIdx.x * blockDim.x;

    if (tid == 0) {
        s_blockNnz = 0;
    }
    if (threadIdx.y == 0 && threadIdx.x <= TILE_SIZE && rowBlockStart + threadIdx.x <= numRowsA) {
        s_rowPtrs[threadIdx.x] = csrA->rowPtrs[rowBlockStart + threadIdx.x];
    }
    __syncthreads();

    // only these threads do any work, but ALL threads must still hit both __syncthreads()
    bool valid = (row < numRowsA && col < numColsB);

    float sum = 0.0f;
    if (valid) {
        if (rowBlockStart + threadIdx.x + 1 <= numRowsA) {
            unsigned int local = threadIdx.x;
            unsigned int rowStart = s_rowPtrs[local];
            unsigned int rowEnd   = s_rowPtrs[local+1];
            unsigned int colStart = cscB->colPtrs[col];
            unsigned int colEnd   = cscB->colPtrs[col+1];
            for (unsigned r = rowStart; r < rowEnd; ++r) {
                unsigned aCol = csrA->colIdxs[r];
                float    aVal = csrA->values[r];
                for (unsigned c = colStart; c < colEnd; ++c) {
                    if (aCol == cscB->rowIdxs[c]) {
                        sum += aVal * cscB->values[c];
                    }
                }
            }
        } else {
            // last partial row in block—fall back to global ptrs
            unsigned int rowStart = csrA->rowPtrs[row];
            unsigned int rowEnd   = csrA->rowPtrs[row+1];
            unsigned int colStart = cscB->colPtrs[col];
            unsigned int colEnd   = cscB->colPtrs[col+1];
            for (unsigned r = rowStart; r < rowEnd; ++r) {
                unsigned aCol = csrA->colIdxs[r];
                float    aVal = csrA->values[r];
                for (unsigned c = colStart; c < colEnd; ++c) {
                    if (aCol == cscB->rowIdxs[c]) {
                        sum += aVal * cscB->values[c];
                    }
                }
            }
        }

        if (sum != 0.0f) {
            unsigned pos = atomicAdd(&s_blockNnz, 1);
            if (pos < MAX_BLOCK_NNZ) {
                s_cooEntries[pos].rowIdx = row;
                s_cooEntries[pos].colIdx = col;
                s_cooEntries[pos].value  = sum;
            }
        }
    }

    __syncthreads();  // all threads reach

    // thread-0 computes how many we stored and where to write in global memory
    if (tid == 0) {
        unsigned int produced = s_blockNnz;
        if (produced > MAX_BLOCK_NNZ) {
            produced = MAX_BLOCK_NNZ;
        }
        s_stored   = produced;
        s_blockPos = (produced>0 ? atomicAdd(d_nnz, produced) : 0u);
    }

    __syncthreads();

    for (unsigned i = tid; i < s_stored; i += blockDim.x * blockDim.y) {
        unsigned outIdx = s_blockPos + i;
        if (outIdx < cooC->capacity) {
            cooC->rowIdxs[outIdx] = s_cooEntries[i].rowIdx;
            cooC->colIdxs[outIdx] = s_cooEntries[i].colIdx;
            cooC->values[outIdx]  = s_cooEntries[i].value;
        }
    }
}

void spmspm_gpu1(COOMatrix* cooMatrix3_h,
                 CSRMatrix* csrMatrix1_d,
                 CSCMatrix*,
                 COOMatrix*,
                 CSRMatrix*,
                 CSCMatrix* cscMatrix2_d,
                 COOMatrix* cooMatrix3_d,
                 unsigned int numRows1,
                 unsigned int,
                 unsigned int numCols2,
                 unsigned int,
                 unsigned int)
{
    unsigned int* d_nnz;
    unsigned int  zero = 0;
    CUDA_ERROR_CHECK(cudaMalloc(&d_nnz, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_nnz, &zero, sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((numRows1 + block.x - 1) / block.x,
                    (numCols2 + block.y - 1) / block.y);

    spmspm_kernel2<<<grid, block>>>(csrMatrix1_d,
                                    cscMatrix2_d,
                                    cooMatrix3_d,
                                    d_nnz,
                                    numRows1,
                                    numCols2);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    unsigned int host_nnz = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&host_nnz, d_nnz,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    if (cooMatrix3_h) {
        cooMatrix3_h->numNonzeros = host_nnz;
    }
    CUDA_ERROR_CHECK(cudaMemcpy(&cooMatrix3_d->numNonzeros,
                               &host_nnz,
                               sizeof(unsigned int),
                               cudaMemcpyHostToDevice));
    cudaFree(d_nnz);
}
