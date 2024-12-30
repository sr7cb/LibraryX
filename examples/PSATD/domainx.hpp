#include <iostream>
#include <vector>
template<typename T>
class NDTensor{
    private:
    int rank;
    int size;
    std::vector<int> dims;
    int stride;
    T** values;

    public:

        NDTensor() {
            rank = 1;
            dims.push_back(1);
            size = 1;
            values = new T*[rank];
            for(int i = 0; i < rank; i++) {
                values[i] = new T[size];
            }
        }

        NDTensor(int r, std::vector<std::vector<int>> in_dims = {1,2,3}) {
            rank = r;
            values = new T*[rank];
            size = -1;
            stride = in_dims.at(0).size();
            for(int i = 0; i < in_dims.size(); i++) {
                int lsize = 1;
                for(int j = 0; j < in_dims.at(i).size(); j++){
                    dims.push_back(in_dims.at(i).at(j));
                    lsize *= in_dims.at(i).at(j);
                }
                values[i] = new T[lsize];
            }
            
            
        }

        NDTensor(int r, std::vector<int> in_dims = {1,2,3}) {
            rank = r;
            values = new T*[rank];
            dims = in_dims;
            stride = in_dims.size();
            size = 1;
            for(int i = 0; i < in_dims.size(); i++) {
                size *= in_dims[i];
            }
            for(int i = 0; i < rank; i++) {
                values[i] = new T[size];
            }
        }
        void fillRandom() {
            for(int i = 0; i < rank; i++) {
                if(size != -1) {
                    for(int j = 0; j < size; j++) {
                        values[i][j] = rand() % 100;
                    }
                } else {
                    for(int j = 0; j < dims.at(i*stride) * dims.at(i*stride+1) * dims.at(i*stride +2); j++) {
                        values[i][j] = rand() % 100;
                    }
                }
            }
        }
        void buildTensor();
        T* getField(int n);
        T* getPencil(int i, int j, int k);
        int getSize(int n);
        int getTotalSize() {return size;}
        int getRank() {return rank;}
        std::vector<int> getDims() {return dims;}
        void setPencil(T *pencil, int i, int j, int k);
        T** data();
};


template<typename T>
int NDTensor<T>::getSize(int n) {
    if(n > dims.size()) {
        std::cout << "size out of range" << std::endl;
    }else {
        return dims.at(n);
    }
    return dims.at(n);
}

template<typename T>
T* NDTensor<T>::getField(int n) {
    // if(n == 0)
    //     return values.data();
    // else {
    //     int lsize = 0;
    //     for(int i = 0; i < n; i++)
    //         lsize += size;
    //     return values.data() + lsize;
    // }
    return values[n];

}
template<typename T>
T* NDTensor<T>::getPencil(int i, int j, int k) {
    if(dims.size() > 3){
        std::cout << "error getPencil doesnt work for irregularly shaped tensors" << std::endl;
        exit(-1);
    }
    std::vector<T> pencil;
    if(i*j*k > size){
        std::cout << "point is greater than size" << std::endl;
    } else{
        for(int l = 0; l < rank; l++) {
            pencil.push_back(values[l][i*dims.at(0)*dims.at(1) +j *dims.at(1) + k]);
        }
    }
    return pencil.data();
}

template<typename T>
void NDTensor<T>::setPencil(T *pencil, int i, int j, int k) {
    if(dims.size() > 3){
        std::cout << "error setPencil doesnt work for irregularly shaped tensors" << std::endl;
        exit(-1);
    }
    if(i*dims.at(0)*dims.at(1) +j *dims.at(1) + k > size){
        std::cout << "point is greater than size" << std::endl;
    } else{
        for(int l = 0; l < rank; l++) {
            values[l][i*dims.at(0)*dims.at(1) +j *dims.at(1) + k] = pencil[l];
        }
    }
}

template<typename T>
void NDTensor<T>::buildTensor(){
    if(dims.empty()) {
        std::cout << "cant initialize empty tensor" << std::endl;
    }else if(size == -1) {
        std::cout << "irregular tensor" << std::endl;
        for(int i = 0; i < rank; i++) {
            for(int j = 0; j < dims.at(i*stride) * dims.at(i*stride+1)*dims.at(i*stride+2); j++) {
                values[i][j] = 1;
            }
        }
    }else {
        for(int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        for(int i = 0; i < rank; i++) {
            for(int j = 0; j < size; j++) {
                values[i][j] = 1;
            }
        }
    }
}

template<typename T>
T** NDTensor<T>::data() {
    return values;
}


class CSRMat{
    public:
    std::vector<int> rows;
    std::vector<int> cols; 
    std::vector<std::complex<double>> vals;
    CSRMat() {}
    CSRMat(std::vector<int> r, std::vector<int> c, std::vector<std::complex<double>> v) : rows(r), cols(c), vals(v) {}
};