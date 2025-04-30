#include <stdio.h>
#include "common.h"
#include "matrix.h"

// Binary search helper function to find first occurrence of a value in a sorted array
int binarySearchFirst(unsigned int* array, int size, unsigned int target) {
    int left = 0;
    int right = size - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (array[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching left for first occurrence
        } else if (array[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

void spmspm_cpu1(COOMatrix *cooMatrix1,
                 CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1,
                 COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2,
                 CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3)
{
    sortCOOMatrix(cooMatrix2);

    float *matrix = (float *)calloc(cooMatrix3->numRows * cooMatrix3->numCols, sizeof(float));

    for (unsigned int i = 0; i < cooMatrix1->numNonzeros; ++i) {
        unsigned int col_i = cooMatrix1->colIdxs[i];
        
        // Binary search for the first occurrence of col_i in cooMatrix2->rowIdxs
        int j = binarySearchFirst(cooMatrix2->rowIdxs, cooMatrix2->numNonzeros, col_i);
        
        // If found, process all matching elements consecutively
        if (j != -1) {
            while (j < cooMatrix2->numNonzeros && cooMatrix2->rowIdxs[j] == col_i) {
                matrix[cooMatrix1->rowIdxs[i] * cooMatrix3->numCols + cooMatrix2->colIdxs[j]] += 
                    cooMatrix1->values[i] * cooMatrix2->values[j];
                j++;
            }
        }
    }

    for (unsigned int i = 0; i < cooMatrix3->numRows; ++i) {
        for (unsigned int j = 0; j < cooMatrix3->numCols; ++j) {
            float val = matrix[i * cooMatrix3->numCols + j];
            if (val != 0.0f) {
                cooMatrix3->rowIdxs[cooMatrix3->numNonzeros] = i;
                cooMatrix3->colIdxs[cooMatrix3->numNonzeros] = j;
                cooMatrix3->values[cooMatrix3->numNonzeros] = val;
                cooMatrix3->numNonzeros++;
            }
        }
    }
}