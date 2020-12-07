#ifndef __NN_HPP
#define __NN_HPP

#include "matrix.hpp"
#include "fact.hpp"
#include "m_sl.hpp"
#include "layer.hpp"

#include <vector>

template <typename T>
class nn {
private:
    double					lr;

	layer<T>				** Layers_arch;
	std::vector<unsigned>	Layers;
	std::vector<unsigned>	Layers_fact;

public:
    nn();

    ~nn();

    void save_coefs();
    void load_coefs();

	void add(unsigned L_size, unsigned fact);

	void compile();

	void set_lr(double learning_rate);

    matrix<T> * query(matrix<T> * I);

    void train(matrix<T> * I, matrix<T> * Tgt);
};

template <typename T>
nn<T>::nn() {};

template<typename T>
void nn<T>::save_coefs() {
	for (unsigned i = 0; i < Layers.size() - 1; i++)
		m_sl<T>::save(*Layers_arch[i]->WM, "", "W" + std::to_string(i));
}

template<typename T>
void nn<T>::load_coefs() {
	for (unsigned i = 0; i < Layers.size() - 1; i++)
		m_sl<T>::load(*Layers_arch[i]->WM, "", "W" + std::to_string(i));
}

template<typename T>
void nn<T>::add(unsigned L_size, unsigned fact) {
	Layers.push_back(L_size);
	Layers_fact.push_back(fact);
}

template<typename T>
void nn<T>::compile() {
	Layers_arch = new layer<T> *[Layers.size() - 1];

	for (unsigned i = 0; i < Layers.size()-1; i++) {
		Layers_arch[i] = new layer<T>(Layers[i], Layers[i + 1], Layers_fact[i+1]);
		Layers_arch[i]->WM->normal();
	}
}

template<typename T>
void nn<T>::set_lr(double learning_rate) {
	this->lr = learning_rate;
}

template<typename T>
matrix<T> * nn<T>::query(matrix<T>* I)
{
	for (unsigned i = 0; i < Layers.size() - 1; i++) {
		(*Layers_arch[i]).IM = i == 0 ? I : (*Layers_arch[i-1]).OM;

		*Layers_arch[i]->OM = matrix<T>::dot(*Layers_arch[i]->WM, *Layers_arch[i]->IM);

		Layers_arch[i]->f_act(*Layers_arch[i]->OM);
	}

	return Layers_arch[2]->OM;
}

template<typename T>
void nn<T>::train(matrix<T>* I, matrix<T>* Tgt) {
	for (unsigned i = 0; i < Layers.size() - 1; i++) {
		(*Layers_arch[i]).IM = i == 0 ? I : (*Layers_arch[i - 1]).OM;

		*Layers_arch[i]->OM = matrix<T>::dot(*Layers_arch[i]->WM, *Layers_arch[i]->IM);

		Layers_arch[i]->f_act(*Layers_arch[i]->OM);
	}

	for (unsigned i = ( Layers.size() - 1 ); i > 0; i--) {
		*Layers_arch[i-1]->EM = (i == (Layers.size() - 1)) ? (*Tgt) - *Layers_arch[i-1]->OM : matrix<T>::dot(Layers_arch[i]->WM->Tr(), *Layers_arch[i]->EM);
	}

	for (unsigned i = (Layers.size() - 1); i > 0; i--) {
		Layers_arch[i-1]->f_act_(*Layers_arch[i-1]->OM);

		*(*Layers_arch[i-1]).WM += matrix<T>::dot((*Layers_arch[i-1]->EM)*(*Layers_arch[i-1]->OM), (Layers_arch[i-1]->IM->Tr())) * this->lr;
	}
};

template <typename T>
nn<T>::~nn() {
	for (unsigned i = 0; i < Layers.size() - 1; i++) {
		delete Layers_arch[i];
	}
	delete[] Layers_arch;
};

#endif /* __NN_HPP */
