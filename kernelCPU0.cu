
#include "common.h"

void spmspm_cpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) {


    cooMatrix3->numRows = csrMatrix1->numRows;
    cooMatrix3->numCols = csrMatrix2->numCols;
    cooMatrix3->numNonzeros = 0;

    for (unsigned int i = 0; i < csrMatrix1->numRows; i++) {
        unsigned int row_start = csrMatrix1->rowPtrs[i];
        unsigned int row_end = csrMatrix1->rowPtrs[i + 1];

        for (unsigned int j = 0; j < csrMatrix2->numCols; j++) {
            unsigned int col_start = cscMatrix2->colPtrs[j];
            unsigned int col_end = cscMatrix2->colPtrs[j + 1];

            float value = 0.0f;
            unsigned int row_idx = row_start;
            unsigned int col_idx = col_start;

            while (row_idx < row_end && col_idx < col_end) {
                unsigned int col_in_row = csrMatrix1->colIdxs[row_idx];
                unsigned int row_in_col = cscMatrix2->rowIdxs[col_idx];

                if      (col_in_row < row_in_col)   {row_idx++;} 
                else if (col_in_row > row_in_col)   {col_idx++;} 
                else {
                    value += csrMatrix1->values[row_idx] * cscMatrix2->values[col_idx];
                    row_idx++;
                    col_idx++;
                }
            }

            // In case the result is zero, we wont add it
            if (value != 0.0f) {
                cooMatrix3->rowIdxs[cooMatrix3->numNonzeros] = i;
                cooMatrix3->colIdxs[cooMatrix3->numNonzeros] = j;
                cooMatrix3->values[cooMatrix3->numNonzeros] = value;
                cooMatrix3->numNonzeros++;
            } 
        }
    }
}



