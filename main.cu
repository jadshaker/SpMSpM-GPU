#include <assert.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "common.h"
#include "matrix.h"
#include "timer.h"

void verify(COOMatrix* cooMatrix, COOMatrix* cooMatrix_ref, unsigned int quickVerify);

void (*spmspm_cpu[])(COOMatrix*, CSRMatrix*, CSCMatrix*,
                     COOMatrix*, CSRMatrix*, CSCMatrix*, COOMatrix*) = {
    spmspm_cpu0,
    spmspm_cpu1
};

void (*spmspm_gpu[])(COOMatrix*, CSRMatrix*, CSCMatrix*,
                     COOMatrix*, CSRMatrix*, CSCMatrix*, COOMatrix*,
                     unsigned int, unsigned int, unsigned int,
                     unsigned int, unsigned int) = {
    spmspm_gpu0,
    spmspm_gpu1,
    spmspm_gpu2,
    spmspm_gpu3,
    spmspm_gpu4,
    spmspm_gpu5,
    spmspm_gpu6,
    spmspm_gpu7,
    spmspm_gpu8,
    spmspm_gpu9
};

int main(int argc, char** argv)
{
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    setbuf(stdout, NULL);

    // ───────────────────────────────── argument parsing
    const char* dataset = "data/dataset1";
    unsigned int runCPUVersion[2]  = {0};
    unsigned int runGPUVersion[10] = {0};
    unsigned int useGPU = 0, quickVerify = 1, writeResult = 0;

    int opt;
    while ((opt = getopt(argc, argv, "d:ab0123456789vw")) >= 0) {
        switch (opt) {
            case 'd': dataset = optarg; break;
            case 'a': runCPUVersion[0] = 1;             break;
            case 'b': runCPUVersion[1] = 1;             break;
            case '0': runGPUVersion[0] = 1; useGPU = 1; break;
            case '1': runGPUVersion[1] = 1; useGPU = 1; break;
            case '2': runGPUVersion[2] = 1; useGPU = 1; break;
            case '3': runGPUVersion[3] = 1; useGPU = 1; break;
            case '4': runGPUVersion[4] = 1; useGPU = 1; break;
            case '5': runGPUVersion[5] = 1; useGPU = 1; break;
            case '6': runGPUVersion[6] = 1; useGPU = 1; break;
            case '7': runGPUVersion[7] = 1; useGPU = 1; break;
            case '8': runGPUVersion[8] = 1; useGPU = 1; break;
            case '9': runGPUVersion[9] = 1; useGPU = 1; break;
            case 'v': quickVerify = 0;      break;
            case 'w': writeResult = 1;      break;
            default:  fprintf(stderr, "\nUnrecognized option!\n"); exit(0);
        }
    }

    // ──────────────────────────────── read input matrices
    std::string matrixFile1 = std::string(dataset) + "-matrix1.txt";
    printf("Reading first input matrix from file: %s\n", matrixFile1.c_str());
    COOMatrix* cooMatrix1 = createCOOMatrixFromFile(matrixFile1.c_str());
    CSRMatrix* csrMatrix1 = createCSRMatrixFromCOOMatrix(cooMatrix1);
    CSCMatrix* cscMatrix1 = createCSCMatrixFromCOOMatrix(cooMatrix1);

    std::string matrixFile2 = std::string(dataset) + "-matrix2.txt";
    printf("Reading second input matrix from file: %s\n", matrixFile2.c_str());
    COOMatrix* cooMatrix2 = createCOOMatrixFromFile(matrixFile2.c_str());
    CSRMatrix* csrMatrix2 = createCSRMatrixFromCOOMatrix(cooMatrix2);
    CSCMatrix* cscMatrix2 = createCSCMatrixFromCOOMatrix(cooMatrix2);

    assert(cooMatrix1->numCols == cooMatrix2->numRows);

    std::string matrixFile3 = std::string(dataset) + "-matrix3.txt";
    printf("Reading reference output matrix from file: %s\n", matrixFile3.c_str());
    COOMatrix* cooMatrix3_ref = createCOOMatrixFromFile(matrixFile3.c_str());

    // ──────────────────────────────── allocate outputs
    printf("Allocating output matrices\n");

    // ---- capacity estimate ---------------------------------------------------
    unsigned long long worstCase = 1ull * csrMatrix1->numRows * csrMatrix2->numCols;
    const unsigned long long MAX_ENTRIES = 256ull * 1024 * 1024;  // ≈3 GiB
    if (worstCase > MAX_ENTRIES) {
        fprintf(stderr,
                "    [warn] worst‑case nnz = %llu > cap %llu; clamping.\n",
                worstCase, MAX_ENTRIES);
        worstCase = MAX_ENTRIES;
    }
    unsigned int capacity = static_cast<unsigned int>(worstCase);
    // -------------------------------------------------------------------------

    COOMatrix* cooMatrix3_cpu =
        createEmptyCOOMatrix(csrMatrix1->numRows, csrMatrix2->numCols, capacity);
    COOMatrix* cooMatrix3_gpu =
        createEmptyCOOMatrix(csrMatrix1->numRows, csrMatrix2->numCols, capacity);

    // ──────────────────────────────── CPU versions
    for (unsigned int version = 0; version < 2; ++version) {
        if (runCPUVersion[version]) {
            printf("Running CPU version %d\n", version);

            clearCOOMatrix(cooMatrix3_cpu);

            Timer timer; startTime(&timer);
            spmspm_cpu[version](cooMatrix1, csrMatrix1, cscMatrix1,
                                cooMatrix2, csrMatrix2, cscMatrix2,
                                cooMatrix3_cpu);
            stopTime(&timer);
            printElapsedTime(timer, "    CPU time", CYAN);

            verify(cooMatrix3_cpu, cooMatrix3_ref, quickVerify);

            if (writeResult) {
                sortCOOMatrix(cooMatrix3_cpu);
                std::string out = std::string(dataset) + "-matrix3-cpu.txt";
                writeCOOMatrixToFile(cooMatrix3_cpu, out.c_str());
            }
        }
    }

    // ──────────────────────────────── GPU versions
    if (useGPU) {
        Timer timer; startTime(&timer);
        COOMatrix* cooMatrix1_d =
            createEmptyCOOMatrixOnGPU(cooMatrix1->numRows,
                                      cooMatrix1->numCols,
                                      cooMatrix1->numNonzeros);
        CSRMatrix* csrMatrix1_d =
            createEmptyCSRMatrixOnGPU(csrMatrix1->numRows,
                                      csrMatrix1->numCols,
                                      csrMatrix1->numNonzeros);
        CSCMatrix* cscMatrix1_d =
            createEmptyCSCMatrixOnGPU(cscMatrix1->numRows,
                                      cscMatrix1->numCols,
                                      cscMatrix1->numNonzeros);

        COOMatrix* cooMatrix2_d =
            createEmptyCOOMatrixOnGPU(cooMatrix2->numRows,
                                      cooMatrix2->numCols,
                                      cooMatrix2->numNonzeros);
        CSRMatrix* csrMatrix2_d =
            createEmptyCSRMatrixOnGPU(csrMatrix2->numRows,
                                      csrMatrix2->numCols,
                                      csrMatrix2->numNonzeros);
        CSCMatrix* cscMatrix2_d =
            createEmptyCSCMatrixOnGPU(cscMatrix2->numRows,
                                      cscMatrix2->numCols,
                                      cscMatrix2->numNonzeros);

        COOMatrix* cooMatrix3_d =
            createEmptyCOOMatrixOnGPU(cooMatrix3_gpu->numRows,
                                      cooMatrix3_gpu->numCols,
                                      cooMatrix3_gpu->capacity);

        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        stopTime(&timer); printElapsedTime(timer, "GPU allocation time");

        // ---------------- copy inputs to GPU
        startTime(&timer);
        copyCOOMatrixToGPU(cooMatrix1, cooMatrix1_d);
        copyCSRMatrixToGPU(csrMatrix1, csrMatrix1_d);
        copyCSCMatrixToGPU(cscMatrix1, cscMatrix1_d);
        copyCOOMatrixToGPU(cooMatrix2, cooMatrix2_d);
        copyCSRMatrixToGPU(csrMatrix2, csrMatrix2_d);
        copyCSCMatrixToGPU(cscMatrix2, cscMatrix2_d);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        stopTime(&timer); printElapsedTime(timer, "Copy to GPU time");

        for (unsigned int version = 0; version < 10; ++version) {
            if (runGPUVersion[version]) {
                printf("Running GPU version %d\n", version);

                clearCOOMatrixOnGPU(cooMatrix3_d);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());

                startTime(&timer);
                spmspm_gpu[version](cooMatrix1_d, csrMatrix1_d, cscMatrix1_d,
                                    cooMatrix2_d, csrMatrix2_d, cscMatrix2_d,
                                    cooMatrix3_d,
                                    cooMatrix1->numRows,
                                    cooMatrix2->numRows,
                                    cooMatrix2->numCols,
                                    cooMatrix1->numNonzeros,
                                    cooMatrix2->numNonzeros);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer); printElapsedTime(timer, "    GPU kernel time", GREEN);

                startTime(&timer);
                copyCOOMatrixFromGPU(cooMatrix3_d, cooMatrix3_gpu);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer); printElapsedTime(timer, "    Copy from GPU time");

                verify(cooMatrix3_gpu, cooMatrix3_ref, quickVerify);

                if (writeResult) {
                    sortCOOMatrix(cooMatrix3_gpu);
                    std::string out = std::string(dataset) + "-matrix3-gpu.txt";
                    writeCOOMatrixToFile(cooMatrix3_gpu, out.c_str());
                }
            }
        }

        // ---------------- free GPU memory
        startTime(&timer);
        freeCOOMatrixOnGPU(cooMatrix1_d);
        freeCSRMatrixOnGPU(csrMatrix1_d);
        freeCSCMatrixOnGPU(cscMatrix1_d);
        freeCOOMatrixOnGPU(cooMatrix2_d);
        freeCSRMatrixOnGPU(csrMatrix2_d);
        freeCSCMatrixOnGPU(cscMatrix2_d);
        freeCOOMatrixOnGPU(cooMatrix3_d);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        stopTime(&timer); printElapsedTime(timer, "GPU deallocation time");
    }

    // ──────────────────────────────── clean‑up
    freeCOOMatrix(cooMatrix1);  freeCSRMatrix(csrMatrix1);  freeCSCMatrix(cscMatrix1);
    freeCOOMatrix(cooMatrix2);  freeCSRMatrix(csrMatrix2);  freeCSCMatrix(cscMatrix2);
    freeCOOMatrix(cooMatrix3_ref);
    freeCOOMatrix(cooMatrix3_cpu);
    freeCOOMatrix(cooMatrix3_gpu);
    return 0;
}

// ─────────────────────────────────────────────────────────────────── verify
void verify(COOMatrix* cooMatrix, COOMatrix* cooMatrix_ref, unsigned int quickVerify)
{
    if (cooMatrix_ref->numNonzeros != cooMatrix->numNonzeros) {
        printf("    \033[1;31mMismatching number of non‑zeros (reference = %d, "
               "computed = %d)\033[0m\n",
               cooMatrix_ref->numNonzeros, cooMatrix->numNonzeros);
        return;
    } else if (quickVerify) {
        printf("    Quick verification succeeded\n");
        printf("        This is not exact; use -v for full check.\n");
    } else {
        printf("    Verifying result\n");
        sortCOOMatrix(cooMatrix);
        for (unsigned int i = 0; i < cooMatrix_ref->numNonzeros; ++i) {
            unsigned int r_ref = cooMatrix_ref->rowIdxs[i];
            unsigned int r_cmp = cooMatrix->rowIdxs[i];
            unsigned int c_ref = cooMatrix_ref->colIdxs[i];
            unsigned int c_cmp = cooMatrix->colIdxs[i];
            float v_ref = cooMatrix_ref->values[i];
            float v_cmp = cooMatrix->values[i];
            if (r_ref != r_cmp || c_ref != c_cmp || fabsf(v_cmp - v_ref) / fabsf(v_ref) > 1e-5) {
                printf("        \033[1;31mMismatch: Ref (%d,%d,%f)  "
                       "Cmp (%d,%d,%f)\033[0m\n",
                       r_ref, c_ref, v_ref, r_cmp, c_cmp, v_cmp);
                return;
            }
        }
        printf("        Verification succeeded\n");
    }
}
