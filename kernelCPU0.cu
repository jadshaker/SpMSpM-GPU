#include "common.h"

void spmspm_cpu0(COOMatrix *cooMatrix1,
                 CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1,
                 COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2,
                 CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3)
{
    float *matrix = (float *)calloc(cooMatrix3->numRows * cooMatrix3->numCols, sizeof(float));

    for (unsigned int i = 0; i < cooMatrix1->numNonzeros; ++i)
        for (unsigned int j = 0; j < cooMatrix2->numNonzeros; ++j)
            if (cooMatrix1->colIdxs[i] == cooMatrix2->rowIdxs[j])
                matrix[cooMatrix1->rowIdxs[i] * cooMatrix3->numCols + cooMatrix2->colIdxs[j]] += cooMatrix1->values[i] * cooMatrix2->values[j];

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