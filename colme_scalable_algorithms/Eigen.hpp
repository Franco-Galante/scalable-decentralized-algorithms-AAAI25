#include <iostream>
#include <vector>
#include <cmath>

namespace Eigen {

    void print_matrix(const std::vector<std::vector<float>>& matrix) {
        for (const auto& row : matrix) {
            for (float value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }
    }

    std::vector<std::vector<float>> matrix_multiply(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
        int m = A.size();
        int n = A[0].size();
        int p = B[0].size();

        std::vector<std::vector<float>> result(m, std::vector<float>(p, 0.0));

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                for (int k = 0; k < n; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();

        std::vector<std::vector<float>> result(n, std::vector<float>(m, 0.0f));

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    // ompute eigenvalues using QR algorithm
    std::vector<float> compute_eigenvalues(const std::vector<std::vector<float>>& matrix, int maxIterations = 100) {
        int n = matrix.size();
        std::vector<std::vector<float>> A = matrix;

        float offDiagonalSum = 0.0f;

        for (int iter = 0; iter < maxIterations; ++iter) {

            // QR decomposition
            std::vector<std::vector<float>> Q(n, std::vector<float>(n, 0.0f));
            std::vector<std::vector<float>> R(n, std::vector<float>(n, 0.0f));

            for (int j = 0; j < n; ++j) {
                std::vector<float> v(n, 0.0f);
                for (int i = 0; i < j; ++i) {
                    float dotProduct = 0.0f;
                    for (int k = 0; k < n; ++k) {
                        dotProduct += Q[k][i] * A[k][j];
                    }
                    for (int k = 0; k < n; ++k) {
                        v[k] += dotProduct * Q[k][i];
                    }
                }

                for (int i = 0; i < n; ++i) {
                    v[i] = A[i][j] - v[i];
                }

                float norm = 0.0f;
                for (float value : v) {
                    norm += value * value;
                }
                norm = static_cast<float>(std::sqrt(norm));

                for (int i = 0; i < n; ++i) {
                    Q[i][j] = v[i] / norm;
                }

            }

            A = matrix_multiply(transpose(Q), matrix_multiply(A, Q));

            // check for convergence (off-diagonal elements close to zero)
            offDiagonalSum = 0.0f;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        offDiagonalSum += std::abs(A[i][j]);
                    }
                }
            }

            if (offDiagonalSum < 1e-10f) {
                std::cout << "INFO: the algorithm converged!" << std::endl;
                break;
            }
        }

        // take eigenvalues from the diagonal of A
        std::vector<float> eigenvalues(n);
        for (int i = 0; i < n; ++i) {
            eigenvalues[i] = A[i][i];
        }

        std::cout << "off diagonal sum: " << offDiagonalSum << std::endl;

        return eigenvalues;
    }

}
