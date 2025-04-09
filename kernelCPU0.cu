
#include "common.h"

void spmspm_cpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) {

    // Initializing some data for the resulting matrix (matrix 3) 
    cooMatrix3->numRows = csrMatrix1->numRows;
    cooMatrix3->numCols = csrMatrix2->numCols;
    cooMatrix3->numNonzeros = 0;

    // Just like in ordinary matrix multiplication, we will go through
    // all the rows of the first matrix. For each row, we will go through all the
    // columns of the second matrix. Thus, it is convenient to use the CSR format for
    // matrix 1 and the CSC format for matrix 2.
    for (unsigned int i = 0; i < csrMatrix1->numRows; i++) {
        // Getting the range in which the values reside
        // in the values array (matrix 1) for each row.
        unsigned int rowStart = csrMatrix1->rowPtrs[i];
        unsigned int rowEnd = csrMatrix1->rowPtrs[i + 1];

        for (unsigned int j = 0; j < cscMatrix2->numCols; j++) {
            // Getting the range in which the values reside
            // in the values array (matrix 2) for each column.
            unsigned int colStart = cscMatrix2->colPtrs[j];
            unsigned int colEnd = cscMatrix2->colPtrs[j + 1];

            float value = 0.0f;

            // Now, we have a row and column, what is left is to 
            // multiply these non-values together. The only pairs we 
            // will multiply are the ones that have matching row index and
            // column index.

            // Starting indexes for matrix 1 row and 
            // matrix 2 column
            unsigned int rowIdx = rowStart;
            unsigned int colIdx = colStart;
            
            // Loop through the row of matrix 1 and the column of matrix 2
            // until we reach the end of either of them
            while (rowIdx < rowEnd && colIdx < colEnd) {
                unsigned int matrix1_cur_col = csrMatrix1->colIdxs[rowIdx];
                unsigned int matrix2_cur_row = cscMatrix2->rowIdxs[colIdx];

                if      (matrix1_cur_col < matrix2_cur_row)   {rowIdx++;} 
                else if (matrix1_cur_col > matrix2_cur_row)   {colIdx++;} 
                else {
                    value += csrMatrix1->values[rowIdx] * cscMatrix2->values[colIdx];
                    rowIdx++;
                    colIdx++;
                }
            }

            // Don't know if this condition is needed, but in case, we added
            // this. (CHECK BEFORE SUMBITTING)
            if (value != 0.0f) {
                // Each value is added to the result matrix in position
                // (row idx, column idx) of matrix 1 and 2 repectively.
                cooMatrix3->rowIdxs[cooMatrix3->numNonzeros] = i;
                cooMatrix3->colIdxs[cooMatrix3->numNonzeros] = j;
                cooMatrix3->values[cooMatrix3->numNonzeros] = value;
                cooMatrix3->numNonzeros++;
            } 
        }
    }
}



