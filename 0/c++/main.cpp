/*
*  File            :   main.cpp
*  Autor           :   Vlasov D.V.
*  Data            :   2020.11.
*  Language        :   c++
*  Description     :
*  Copyright(c)    :   2020 Vlasov D.V.
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <chrono>
#include "matrix.hpp"
#include "nn.hpp"
#include "fact.hpp"

bool read_csv_line(std::ifstream *fp, int * label, matrix<double> * M) {
    std::string line;
    int val;
    int index = 0;
    if (!std::getline(*fp, line))
        return false;
    std::stringstream ss(line);
    ss >> *label;
    if (ss.peek() == ',')
        ss.ignore();
    while (ss >> val) {
        M->set_val(index, 0, val);
        index++;
        if (ss.peek() == ',')
            ss.ignore();
    }
    return true;
}

int main(int argc, char ** argv) {
    nn<double>      nn_mnist;
    matrix<double>  * O_;
    matrix<double>  I_(784, 1);
    matrix<double>  T_(10, 1);
    int             label;
    std::ifstream   fp;
    std::string     path2folder;

    path2folder += argv[1];

    int progress = 0;

    long time_c;
    long time_n;

    long iter;

    nn_mnist.add(784, ReLU_e);
    nn_mnist.add(32, ReLU_e);
    nn_mnist.add(32, ReLU_e);
    nn_mnist.add(10, sigmoida_e);

    nn_mnist.set_lr(0.001);

    nn_mnist.compile();

    //nn_mnist.load_coefs();

    for (int ep = 0; ep < 7; ep++) {
        std::cout << "Epoch : " << ep << std::endl;
        std::cout << "Train model : " << std::endl;
        progress = 0;
        fp = std::ifstream(path2folder + "mnist_train.csv");
        iter = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        time_c = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        while (read_csv_line(&fp, &label, &I_)) {
            T_.fill_matrix(0.001);
            T_.set_val(label, 0, 0.99);

            I_ /= 255.0;
            I_ *= 0.99;
            I_ += 0.01;

            nn_mnist.train(&I_, &T_);
            progress++;
            time_n = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if ((time_n - time_c) > 200) {
                time_c = time_n;
                std::cout << "progress : " << (progress / 60000.0) * 100.0 << '\r';
            }
        }
        iter = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - iter;
        std::cout << "progress : " << 100.0 << "                         " << '\r';
        std::cout << "iteration time = " << iter << " ms";
        fp.close();

        std::cout << std::endl;

        nn_mnist.save_coefs();

        fp = std::ifstream(path2folder + "mnist_test.csv");
        std::cout << "Test model : " << std::endl;
        unsigned pass_cnt = 0;
        progress = 0;
        iter = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        time_c = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        while (read_csv_line(&fp, &label, &I_)) {
            int max_index = 0;
            double max_index_v;
            T_.fill_matrix(0.01);
            T_.set_val(label, 0, 0.99);

            I_ /= 255.0;
            I_ *= 0.99;
            I_ += 0.01;

            O_ = nn_mnist.query(&I_);

            max_index_v = O_->get_val(0, 0);
            for (unsigned i = 0; i<10; i++) {
                if (max_index_v < O_->get_val(i, 0)) {
                    max_index_v = O_->get_val(i, 0);
                    max_index = i;
                }
            }

            if (max_index == label)
                pass_cnt++;

            progress++;
            time_n = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if ((time_n - time_c) > 200) {
                time_c = time_n;
                std::cout << "progress : " << (progress / 10000.0) * 100.0 << '\r';
            }
        }
        iter = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - iter;
        std::cout << "progress : " << 100.0 << "                         " << '\r';
        std::cout << "iteration time = " << iter << " ms";
        fp.close();

        std::cout << std::endl;

        std::cout << "" << pass_cnt / 10000.0 << std::endl;
    }

    std::cout << std::endl;

    nn_mnist.save_coefs();

    return 0;
}
