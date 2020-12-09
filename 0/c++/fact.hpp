#ifndef __FACT_HPP
#define __FACT_HPP

#include <cmath>

template <typename T>
class matrix;

template<typename T>
void sigmoida(matrix<T> & M) {
    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++)
            M.data[i][j] = 1.0 / (exp(-M.data[i][j]) + 1);
}

template<typename T>
void sigmoida_(matrix<T> & M) {
    T val;
    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++) {
            val = (1.0 / (exp(-M.data[i][j]) + 1));
            M.data[i][j] = val * (1-val);
        }
}

template<typename T>
void ReLU(matrix<T> & M) {
    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++)
            M.data[i][j] = M.data[i][j] >= 0.0 ? M.data[i][j] : 0.0;
}

template<typename T>
void ReLU_(matrix<T> & M) {
    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++)
            M.data[i][j] = (M.data[i][j] >= 0.0 ? 1.0 : 0.0);
}

template<typename T>
void softmax(matrix<T> & M) {
    T sum = 0;
    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++)
            sum += exp(M.data[i][j]);

    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++)
            M.data[i][j] = exp(M.data[i][j]) / sum;
}

template<typename T>
void softmax_(matrix<T> & M) {
    T sum = 0;
    T val;
    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++)
            sum += exp(M.data[i][j]);

    for (int i = 0; i < M.size_i; i++)
        for (int j = 0; j < M.size_j; j++) {
            val = (exp(M.data[i][j]) / sum);
            M.data[i][j] = val*(1-val);
        }
}

enum fact_e {
    sigmoida_e,
    ReLU_e,
    softmax_e
};

#endif /* __FACT_HPP */
