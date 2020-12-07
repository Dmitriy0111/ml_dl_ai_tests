#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "matrix.hpp"
#include "fact.hpp"

template <typename T>
class layer {
public:
    matrix<T>       * IM;
    matrix<T>       * WM;
	matrix<T>       * OM;
	matrix<T>       * EM;

	void			(*f_act)(matrix<T> &);
	void			(*f_act_)(matrix<T> &);

	layer();
	layer(unsigned IM_size, unsigned OM_size, unsigned fact);

	~layer();
};



template<typename T>
layer<T>::layer() {}

template<typename T>
layer<T>::layer(unsigned IM_size, unsigned OM_size, unsigned fact) {
	WM = new matrix<T>(OM_size, IM_size);
	OM = new matrix<T>(OM_size, 1);
	EM = new matrix<T>(OM_size, 1);
	switch (fact)
	{
	case sigmoida_e:
		f_act = sigmoida;
		f_act_ = sigmoida_;
		break;
	case ReLU_e:
		f_act = ReLU;
		f_act_ = ReLU_;
		break;
	case softmax_e:
		f_act = softmax;
		f_act_ = softmax_;
		break;
	default:
		break;
	}
}

template<typename T>
layer<T>::~layer() {
	delete WM;
	delete OM;
	delete EM;
}

#endif /* __LAYER_HPP */
