#include "common.h"

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
    COOMatrix* cooC,
    unsigned int* d_nnz, 
    unsigned int     numRowsA,
    unsigned int     numColsB)
{
    
    __shared__ unsigned int s_rowPtrs[TILE_SIZE + 1];
    __shared__ COOEntry  s_cooEntries[MAX_BLOCK_NNZ];
    __shared__ unsigned int s_blockNnz;
    __shared__ unsigned int s_stored;
    __shared__ unsigned int s_blockPos;

    
    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int rowBlockStart = blockIdx.x * blockDim.x; 

    
    if (tid == 0) {
        s_blockNnz = 0;
    }

    
    if (threadIdx.y == 0 && threadIdx.x < TILE_SIZE) { 
        unsigned int loadIdx = rowBlockStart + threadIdx.x;
        if (loadIdx <= numRowsA) { 
             s_rowPtrs[threadIdx.x] = csrA->rowPtrs[loadIdx];
        }
         
    }

    
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        unsigned int loadIdx = rowBlockStart + TILE_SIZE;
        s_rowPtrs[TILE_SIZE] = (loadIdx <= numRowsA)
                               ? csrA->rowPtrs[loadIdx]
                               : csrA->rowPtrs[numRowsA]; 
    }
    __syncthreads(); 
    
    bool valid = (row < numRowsA && col < numColsB);
    float sum = 0.0f;

    if (valid) {
        if (row + 1 <= numRowsA) { 
            unsigned int localRowIdx = threadIdx.x;
            unsigned int rowStartA = s_rowPtrs[localRowIdx];
            unsigned int rowEndA   = s_rowPtrs[localRowIdx + 1]; 
            unsigned int colStartB = cscB->colPtrs[col];
            unsigned int colEndB   = cscB->colPtrs[col + 1]; 

            for (unsigned int i = rowStartA; i < rowEndA; ++i) {
                unsigned int aCol = csrA->colIdxs[i];
                float        aVal = csrA->values[i];
                for (unsigned int j = colStartB; j < colEndB; ++j) {
                    if (aCol == cscB->rowIdxs[j]) {
                        sum += aVal * cscB->values[j];
                    }
                }
            }
        } else {
            
            unsigned int rowStartA = csrA->rowPtrs[row];
            unsigned int rowEndA   = csrA->rowPtrs[row + 1]; 
            unsigned int colStartB = cscB->colPtrs[col];
            unsigned int colEndB   = cscB->colPtrs[col + 1];

            for (unsigned int i = rowStartA; i < rowEndA; ++i) {
                unsigned int aCol = csrA->colIdxs[i];
                float        aVal = csrA->values[i];
                for (unsigned int j = colStartB; j < colEndB; ++j) {
                    if (aCol == cscB->rowIdxs[j]) {
                        sum += aVal * cscB->values[j];
                    }
                }
            }
        }

        if (sum != 0.0f) {
            unsigned int pos = atomicAdd(&s_blockNnz, 1);
            if (pos < MAX_BLOCK_NNZ) {
                s_cooEntries[pos].rowIdx = row;
                s_cooEntries[pos].colIdx = col;
                s_cooEntries[pos].value  = sum;
            }
        }
    }

    __syncthreads();

    
    if (tid == 0) {
        unsigned int produced = s_blockNnz;
        if (produced > MAX_BLOCK_NNZ) {
            produced = MAX_BLOCK_NNZ;
        }
        s_stored = produced;
        s_blockPos = (produced > 0) ? atomicAdd(d_nnz, produced) : 0u;

        
        unsigned int capacity = cooC->capacity;
        if (s_stored > 0) {
           if (s_blockPos >= capacity) {
               s_stored = 0;
           } else if (s_blockPos + s_stored > capacity) {
               s_stored = capacity - s_blockPos;
           }
        }
    }
    __syncthreads();

   
    for (unsigned int i = tid; i < s_stored; i += blockDim.x * blockDim.y) {
        unsigned int outIdx = s_blockPos + i;
        if (outIdx < cooC->capacity) {
             cooC->rowIdxs[outIdx] = s_cooEntries[i].rowIdx;
             cooC->colIdxs[outIdx] = s_cooEntries[i].colIdx;
             cooC->values[outIdx]  = s_cooEntries[i].value;
        }
    }
} 

void spmspm_gpu2(COOMatrix* cooMatrix3_h,
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
                 unsigned int
                 )
{
    unsigned int* d_nnz;
    unsigned int  zero = 0;

    CUDA_ERROR_CHECK(cudaMalloc((void**)&d_nnz, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_nnz, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
    CUDA_ERROR_CHECK(cudaMemcpy(&host_nnz, d_nnz, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK(cudaMemcpy(&(cooMatrix3_d->numNonzeros),
                                &host_nnz,
                                sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    CUDA_ERROR_CHECK(cudaFree(d_nnz));
}