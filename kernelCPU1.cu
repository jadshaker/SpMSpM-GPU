#include "common.h"

void spmspm_cpu1(COOMatrix *cooMatrix1,
                 CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1,
                 COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2,
                 CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3)
{
    // CSC * CSC implementation
    float *col = (float *)calloc(cooMatrix3->numRows, sizeof(float));

    for (unsigned int colB = 0; colB < cscMatrix2->numCols; ++colB)
    {
        memset(col, 0, cooMatrix3->numRows * sizeof(float));

        unsigned int colStart2 = cscMatrix2->colPtrs[colB];
        unsigned int colEnd2 = cscMatrix2->colPtrs[colB + 1];

        for (unsigned int i = colStart2; i < colEnd2; ++i)
        {
            float valB = cscMatrix2->values[i];
            unsigned int colA = cscMatrix2->rowIdxs[i];

            unsigned int colStart1 = cscMatrix1->colPtrs[colA];
            unsigned int colEnd1 = cscMatrix1->colPtrs[colA + 1];

            for (unsigned int j = colStart1; j < colEnd1; ++j)
            {
                unsigned int rowA = cscMatrix1->rowIdxs[j];
                float valA = cscMatrix1->values[j];
                col[rowA] += valA * valB;
            }
        }

        for (unsigned int rowIdx = 0; rowIdx < cooMatrix3->numRows; ++rowIdx)
        {
            if (col[rowIdx] != 0)
            {
                cooMatrix3->rowIdxs[cooMatrix3->numNonzeros] = rowIdx;
                cooMatrix3->colIdxs[cooMatrix3->numNonzeros] = colB;
                cooMatrix3->values[cooMatrix3->numNonzeros] = col[rowIdx];
                cooMatrix3->numNonzeros++;
            }
        }
    }
}