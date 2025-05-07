#include "common.h"

void spmspm_cpu0(COOMatrix *cooMatrix1,
                 CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1,
                 COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2,
                 CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3)
{
    // CSR * CSR implementation
    float *row = (float *)calloc(cooMatrix3->numCols, sizeof(float));

    for (unsigned int rowA = 0; rowA < csrMatrix1->numRows; ++rowA)
    {
        memset(row, 0, cooMatrix3->numCols * sizeof(float));

        unsigned int rowStart1 = csrMatrix1->rowPtrs[rowA];
        unsigned int rowEnd1 = csrMatrix1->rowPtrs[rowA + 1];

        for (unsigned int i = rowStart1; i < rowEnd1; ++i)
        {
            float valA = csrMatrix1->values[i];
            unsigned int rowB = csrMatrix1->colIdxs[i];

            unsigned int rowStart2 = csrMatrix2->rowPtrs[rowB];
            unsigned int rowEnd2 = csrMatrix2->rowPtrs[rowB + 1];

            for (unsigned int j = rowStart2; j < rowEnd2; ++j)
            {
                unsigned int colB = csrMatrix2->colIdxs[j];
                float valB = csrMatrix2->values[j];
                row[colB] += valA * valB;
            }
        }

        for (unsigned int colIdx = 0; colIdx < cooMatrix3->numCols; ++colIdx)
        {
            if (row[colIdx] != 0)
            {
                cooMatrix3->rowIdxs[cooMatrix3->numNonzeros] = rowA;
                cooMatrix3->colIdxs[cooMatrix3->numNonzeros] = colIdx;
                cooMatrix3->values[cooMatrix3->numNonzeros] = row[colIdx];
                cooMatrix3->numNonzeros++;
            }
        }
    }
}