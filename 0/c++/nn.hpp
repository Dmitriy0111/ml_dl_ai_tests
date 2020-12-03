#ifndef __NN_HPP
#define __NN_HPP

#include "matrix.hpp"
#include <fstream>

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
    matrix<T>       * HE;
    matrix<T>       * who;
    matrix<T>       * O;
    matrix<T>       * OE;

public:
    T               (*act_f)(T);

    nn();
    nn(unsigned I_len, unsigned H_len, unsigned O_len, double learning_rate);

    ~nn();

    void cout_nn();

    void save_coefs();
    void load_coefs();

    matrix<T> * query(matrix<T> * I);
    void train(matrix<T> * I, matrix<T> * T);
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
void nn<T>::cout_nn() {
    I->cout_matrix();
    wih->cout_matrix();
    H->cout_matrix();
    who->cout_matrix();
    O->cout_matrix();
}

template<typename T>
void nn<T>::save_coefs() {
    std::ofstream fp;

    fp = std::ofstream("wih.dat");
    for (unsigned i = 0; i < wih->get_size_i(); i++) {
        for (unsigned j = 0; j < wih->get_size_j(); j++)
            fp << wih->get_val(i, j) << ( j == wih->get_size_j() - 1 ) ? "" : " ";
        fp << '\n';
    }
    fp.close();

    fp = std::ofstream("who.dat");
    for (unsigned i = 0; i < who->get_size_i(); i++) {
        for (unsigned j = 0; j < who->get_size_j(); j++)
            fp << who->get_val(i, j) << " ";
        fp << '\n';
    }
    fp.close();
}

template<typename T>
void nn<T>::load_coefs() {
    std::ifstream fp;
    std::string line;
    std::stringstream ss;

    unsigned i;
    unsigned j;
    double val;

    fp = std::ifstream("who.dat");
    i = 0;
    j = 0;
    while (true)
    {
        if (!std::getline(fp, line))
            break;

        ss = std::stringstream(line);

        while (ss >> val) {
            who->set_val(j, i, val);
            if (ss.peek() == ' ')
                ss.ignore();
            i++;
        }
        i = 0;
        j++;
    }

    fp.close();

    fp = std::ifstream("wih.dat");
    i = 0;
    j = 0;
    while (true)
    {
        if (!std::getline(fp, line))
            break;

        ss = std::stringstream(line);

        while (!ss.eof()) {
            ss >> val;
            wih->set_val(j, i, val);
            i++;
        }
        i = 0;
        j++;
    }
    
    fp.close();
}

template<typename T>
matrix<T> * nn<T>::query(matrix<T>* I)
{
    this->I = I;

    *H = (*wih) * (*this->I);
    H->apply_f(act_f);

    *O = (*who) * (*H);
    O->apply_f(act_f);

    return O;
}

template<typename T>
void nn<T>::train(matrix<T>* I, matrix<T>* T) {
    this->I = I;

    *H = (*wih) * (*this->I);
    H->apply_f(act_f);

    *O = (*who) * (*H);
    O->apply_f(act_f);

    *OE = (*T) - (*this->O);

    *HE = who->Tr() * (*this->OE);

    matrix<double> datas(*OE);

    for (unsigned i = 0; i < datas.get_size_i(); i++) {
        datas.set_val(i, 0, datas.get_val(i, 0)*(*O).get_val(i, 0));
    }

    for (unsigned i = 0; i < datas.get_size_i(); i++) {
        datas.set_val(i, 0, datas.get_val(i, 0)*( 1.0 - (*O).get_val(i, 0) ));
    }

    datas = datas * H->Tr();
    datas = datas * this->lr;
    *who += datas; // (((*OE) * (*O) * ((*O)*(-1.0) + 1)) * H->Tr()) * this->lr;

    datas = (*HE);

    for (unsigned i = 0; i < datas.get_size_i(); i++) {
        datas.set_val(i, 0, datas.get_val(i, 0)*(*H).get_val(i, 0));
    }

    for (unsigned i = 0; i < datas.get_size_i(); i++) {
        datas.set_val(i, 0, datas.get_val(i, 0)*(1.0 - (*H).get_val(i, 0)));
    }

    datas = datas * I->Tr();
    datas = datas * this->lr;
    *wih += datas; //(((*HE) * (*H) * ((*H)*(-1.0) + 1)) * this->I->Tr()) * this->lr;
};

template <typename T>
nn<T>::~nn() {
    delete wih;
    delete H;
    delete who;
    delete O;
};

#endif /* __NN_HPP */
