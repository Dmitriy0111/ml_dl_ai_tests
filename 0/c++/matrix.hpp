#ifndef __MATRIX_HPP
#define __MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <cassert>

#include "fact.hpp"

template <typename T>
class matrix {
public:
    matrix();
    matrix(unsigned size_i, unsigned size_j, T fill = 0);
    matrix(matrix<T> & oth);
    ~matrix();

    void cout_matrix();

    void fill_matrix(T fill);

    void set_val(int i, int j, T val = 0);
    T get_val(int i, int j);

    unsigned get_size_i();
    unsigned get_size_j();

    void apply_f(T(*f)(T));

    matrix<T> Tr();

    void create_data(int size_i, int size_j);
    void delete_data();
    void copy_data(T ** oth_data);

    void normal();

    static matrix<T> dot(const matrix<T> & a, const matrix<T> & b);

    matrix<T> operator+(const matrix<T> & oth);
    matrix<T> operator+(const T & oth);
    matrix<T> operator+=(const matrix<T> & oth);
    matrix<T> operator+=(const T & oth);

    matrix<T> operator-(const matrix<T> & oth);
    matrix<T> operator-(const T & oth);
    matrix<T> operator-=(const matrix<T> & oth);
    matrix<T> operator-=(const T & oth);

    matrix<T> operator*(const matrix<T> & oth);
    matrix<T> operator*(const T & oth);
    matrix<T> operator*=(const matrix<T> & oth);
    matrix<T> operator*=(const T & oth);

    matrix<T> operator/=(const T & oth);

    matrix<T> operator=(const matrix<T> & oth);

    matrix<T> operator-();

    template <typename T>
    friend void sigmoida(matrix<T> & M);
    template <typename T>
    friend void sigmoida_(matrix<T> & M);
    template <typename T>
    friend void ReLU(matrix<T> & M);
    template <typename T>
    friend void ReLU_(matrix<T> & M);
    template <typename T>
    friend void softmax(matrix<T> & M);
    template <typename T>
    friend void softmax_(matrix<T> & M);

private:
    T           ** data;
    unsigned    size_i;
    unsigned    size_j;
};

template <typename T>
matrix<T>::matrix() {
}

template <typename T>
matrix<T>::matrix(unsigned size_i, unsigned size_j, T fill) {
    this->size_i = size_i;
    this->size_j = size_j;

    create_data(size_i, size_j);

    fill_matrix(fill);
}

template <typename T>
matrix<T>::matrix(matrix<T> & oth) {
    this->size_i = oth.size_i;
    this->size_j = oth.size_j;

    create_data(size_i, size_j);

    copy_data(oth.data);
}

template <typename T>
matrix<T>::~matrix() {
    delete_data();
}

template <typename T>
void matrix<T>::cout_matrix() {
    for (int i = 0; i < size_i; i++) {
        for (int j = 0; j < size_j; j++)
            std::cout << std::setw(7) << data[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void matrix<T>::fill_matrix(T fill) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            data[i][j] = fill;
}

template<typename T>
void matrix<T>::set_val(int i, int j, T val = 0) {
    this->data[i][j] = val;
}

template<typename T>
T matrix<T>::get_val(int i, int j) {
    return this->data[i][j];
}

template<typename T>
unsigned matrix<T>::get_size_i() {
    return this->size_i;
}

template<typename T>
unsigned matrix<T>::get_size_j() {
    return this->size_j;
}

template<typename T>
void matrix<T>::apply_f(T(*f)(T)) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            data[i][j] = f(data[i][j]);
}

template<typename T>
matrix<T> matrix<T>::Tr() {
    matrix<T> tmp(this->size_j, this->size_i);

    for (int i = 0; i < this->size_j; i++)
        for (int j = 0; j < this->size_i; j++)
            tmp.data[i][j] = this->data[j][i];

    return tmp;
}

template<typename T>
void matrix<T>::create_data(int size_i, int size_j) {
    data = new T*[size_i];
    *data = new T[size_i*size_j];

    for (unsigned i = 0; i < size_i; i += 1) {
        data[i] = data[0] + i*size_j;
    }
}

template<typename T>
void matrix<T>::delete_data() {
    if (data != NULL) {
        delete[] * data;
        delete[] data;
    }
}

template<typename T>
void matrix<T>::copy_data(T ** oth_data) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] = oth_data[i][j];
}

template<typename T>
void matrix<T>::normal() {
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, pow(this->size_i, -0.5));

    for (int i = 0; i < this->size_i; i++)
        for (int j = 0; j < this->size_j; j++)
            this->data[i][j] = dist(gen);
}

template<typename T>
matrix<T> matrix<T>::dot(const matrix<T>& a, const matrix<T>& b) {
    assert(a.size_j == b.size_i);

    matrix<T> tmp(a.size_i, b.size_j);

    for (int i = 0; i < tmp.size_i; i++)
        for (int j = 0; j < tmp.size_j; j++)
            for (int k = 0; k < a.size_j; k++)
                tmp.data[i][j] += a.data[i][k] * b.data[k][j];

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator+(const matrix<T> & oth) {
    assert((this->size_i == oth.size_i) && (this->size_j == oth.size_j));

    matrix<T> tmp(*this);

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            tmp.data[i][j] += oth.data[i][j];

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator+(const T & oth) {
    matrix<T> tmp(*this);

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            tmp.data[i][j] += oth;

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator+=(const matrix<T> & oth) {
    assert((this->size_i == oth.size_i) && (this->size_j == oth.size_j));

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] += oth.data[i][j];

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator+=(const T & oth) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] += oth;

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator-(const matrix<T> & oth) {
    assert((this->size_i == oth.size_i) && (this->size_j == oth.size_j));

    matrix<T> tmp(*this);

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            tmp.data[i][j] -= oth.data[i][j];

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator-(const T & oth) {
    matrix<T> tmp(*this);

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            tmp.data[i][j] -= oth;

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator-=(const matrix<T> & oth) {
    assert((this->size_i == oth.size_i) && (this->size_j == oth.size_j));

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] -= oth.data[i][j];

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator-=(const T & oth) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] -= oth;

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator*(const matrix<T> & oth) {
    assert((this->size_i == oth.size_i) && (this->size_j == oth.size_j));

    matrix<T> tmp(this->size_i, this->size_j);

    for (int i = 0; i < tmp.size_i; i++)
        for (int j = 0; j < tmp.size_j; j++)
            tmp.data[i][j] = this->data[i][j] * oth.data[i][j];

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator*(const T & oth) {
    matrix<T> tmp(*this);

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            tmp.data[i][j] = tmp.data[i][j] * oth;

    return tmp;
}

template<typename T>
matrix<T> matrix<T>::operator*=(const matrix<T> & oth) {
    assert((this->size_i == oth.size_i) && (this->size_j == oth.size_j));

    for (int i = 0; i < tmp.size_i; i++)
        for (int j = 0; j < tmp.size_j; j++)
            this->data[i][j] = this->data[i][j] * oth.data[i][j];

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator*=(const T & oth) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] = this->data[i][j] * oth;

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator/=(const T & oth) {
    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            this->data[i][j] /= oth;

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator=(const matrix<T> & oth) {
    if (this != &oth)

        if ((this->size_i != oth.size_i) || (this->size_j != oth.size_j)) {
            if (data != NULL) {
                delete[] data[0];
                delete[] data;
            }
            this->size_i = oth.size_i;
            this->size_j = oth.size_j;
            create_data(size_i, size_j);
        }

    copy_data(oth.data);

    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator-() {
    matrix<T> tmp(*this);

    for (int i = 0; i < size_i; i++)
        for (int j = 0; j < size_j; j++)
            tmp.data[i][j] = -tmp.data[i][j];

    return tmp;
}

#endif /* __MATRIX_HPP */
