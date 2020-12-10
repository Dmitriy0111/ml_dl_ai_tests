/*
*  File            :   m_sl.hpp
*  Autor           :   Vlasov D.V.
*  Data            :   2020.11.
*  Language        :   c++
*  Description     :
*  Copyright(c)    :   2020 Vlasov D.V.
*/

#ifndef __M_SL_HPP
#define __M_SL_HPP

#include <string>
#include <fstream>
#include "matrix.hpp"

template <typename T>
class m_sl {
public:
    static bool save(matrix<T> & M, std::string path2file, std::string file_name);
    static bool load(matrix<T> & M, std::string path2file, std::string file_name);
};

template <typename T>
bool m_sl<T>::save(matrix<T> & M, std::string path2file, std::string file_name) {
    std::ofstream fp;

    fp = std::ofstream(path2file + file_name + ".dat");
    if (!fp) {
        return false;
    }
    for (unsigned i = 0; i < M.get_size_i(); i++) {
        for (unsigned j = 0; j < M.get_size_j(); j++) {
            fp << M.get_val(i, j);
            if (j != (M.get_size_j() - 1))
                fp << " ";
        }
        fp << '\n';
    }

    fp.close();
    return true;
}

template <typename T>
bool m_sl<T>::load(matrix<T> & M, std::string path2file, std::string file_name) {
    std::ifstream fp;
    std::string line;
    std::stringstream ss;

    unsigned i = 0;
    unsigned j = 0;
    T val;

    fp = std::ifstream(path2file + file_name + ".dat");
    if (!fp) {
        return false;
    }
    while (true)
    {
        if (!std::getline(fp, line))
            break;

        ss = std::stringstream(line);

        while (ss >> val) {
            M.set_val(j, i, val);
            if (ss.peek() == ' ')
                ss.ignore();
            i++;
        }
        i = 0;
        j++;
    }

    fp.close();
    return true;
}

#endif /* __M_SL_HPP */
