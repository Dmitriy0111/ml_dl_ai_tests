#ifndef __FACT_HPP
#define __FACT_HPP

#include <cmath>

double sigmoida(double val) {
    return 1.0 / (exp(-val) + 1);
}

double sigmoida_(double val) {
    return sigmoida(val) * (1.0 - sigmoida(val));
}

double ReLU(double val) {
    return (val >= 0.0 ? val : 0.0);
}

double ReLU_(double val) {
    return (val >= 0.0 ? val : 0.0)*(val >= 0.0 ? 1.0 : 0.0);
}

#endif /* __FACT_HPP */
