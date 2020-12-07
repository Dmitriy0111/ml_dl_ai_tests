#ifndef __NN_HPP
#define __NN_HPP

#include "matrix.hpp"
#include "fact.hpp"
#include "m_sl.hpp"

template <typename T>
class nn {
private:
    unsigned        I_len;
    unsigned        H_len;
    unsigned        O_len;

    double          lr;

    matrix<T>       * I;
    matrix<T>       * wih;
    matrix<T>       * H;
    matrix<T>       * who;
    matrix<T>       * O;
    matrix<T>       * HE;
    matrix<T>       * OE;

public:
    nn();
    nn(unsigned I_len, unsigned H_len, unsigned O_len, double learning_rate);

    ~nn();

    void save_coefs();
    void load_coefs();

    matrix<T> * query(matrix<T> * I);
    void train(matrix<T> * I, matrix<T> * Tgt);
};

template <typename T>
nn<T>::nn() {};

template <typename T>
nn<T>::nn(unsigned I_len, unsigned H_len, unsigned O_len, double learning_rate) {
    this->I_len = I_len;
    this->H_len = H_len;
    this->O_len = O_len;

    this->lr = learning_rate;

    wih = new matrix<T>(H_len, I_len);
    wih->normal();
    H = new matrix<T>(H_len, 1);
    HE = new matrix<T>(H_len, 1);
    who = new matrix<T>(O_len, H_len);
    who->normal();
    O = new matrix<T>(O_len, 1);
    OE = new matrix<T>(O_len, 1);
}

template<typename T>
void nn<T>::save_coefs() {
	m_sl<T>::save(*wih, "", "wih");
	m_sl<T>::save(*who, "", "who");
}

template<typename T>
void nn<T>::load_coefs() {
	m_sl<T>::load(*wih, "", "wih");
	m_sl<T>::load(*who, "", "who");
}

template<typename T>
matrix<T> * nn<T>::query(matrix<T>* I)
{
    this->I = I;

    *H = matrix<T>::dot(*wih, *this->I);
    ReLU(*H);

    *O = matrix<T>::dot(*who, *H);
    softmax(*O);

    return O;
}

template<typename T>
void nn<T>::train(matrix<T>* I, matrix<T>* Tgt) {
    this->I = I;

    *H = matrix<T>::dot(*wih, *this->I);
    ReLU(*H);

    *O = matrix<T>::dot(*who, *H);
	softmax(*O);

    *OE = (*Tgt) - (*this->O);

    *HE = matrix<T>::dot(who->Tr(), *this->OE);

	softmax_(*O);
    *who += matrix<T>::dot((*OE)*(*O), H->Tr()) * this->lr;

    ReLU_(*H);
    *wih += matrix<T>::dot((*HE)*(*H), I->Tr()) * this->lr;
};

template <typename T>
nn<T>::~nn() {
    delete wih;
    delete H;
    delete who;
    delete O;
    delete HE;
    delete OE;
};

#endif /* __NN_HPP */
