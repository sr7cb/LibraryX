#include <iostream>
#include <vector>
template<typename T>
class NDTensor{
    private:
    int rank;
    int size;
    std::vector<int> dims;
    std::vector<T> values;

    public:

        NDTensor() {
            rank = 1;
            dims.push_back(1);
            size = 1;
        }
        NDTensor(int r, std::vector<int> in_dims = {1,2,3}) {
                rank = r;
                dims = in_dims;
                size = 1;
                for(int i = 0; i < in_dims.size(); i++) {
                    size *= in_dims[i];
                }
        }
        void fillRandom() {
            for(int i = 0; i < values.size(); i++) {
                values[i] = rand() % 100;
            }
        }
        void buildTensor();
        T* getField(int n);
        T* getPencil(int i, int j, int k);
        int getSize(int n);
        int getTotalSize() {return size;}
        int getRank() {return rank;}
        std::vector<int> getDims() {return dims;}
        std::vector<T> getValues() {return values;}
        void setPencil(T *pencil, int n);
        T* data();
};


template<typename T>
int NDTensor<T>::getSize(int n) {
    if(n > dims.size()) {
        std::cout << "size out of range" << std::endl;
    }else {
        return dims.at(n);
    }
}

template<typename T>
T* NDTensor<T>::getField(int n) {
    if(n == 0)
        return values.data();
    else {
        int lsize = 0;
        for(int i = 0; i < n; i++)
            lsize += size;
        return values.data() + lsize;
    }

}
template<typename T>
T* NDTensor<T>::getPencil(int i, int j, int k) {
    std::vector<T> pencil;
    if(i*j*k > size){
        std::cout << "point is greater than size" << std::endl;
    } else{
        for(int i = 0; i < rank; i++) {
            pencil.push_back(values[i*size + i*j*k]);
        }
    }
    return pencil.data();
}

template<typename T>
void NDTensor<T>::setPencil(T *pencil, int n) {
    if(n > size){
        std::cout << "point is greater than size" << std::endl;
    } else{
        for(int i = 0; i < rank; i++) {
            values[i*size + n] = pencil[i];
        }
    }
}

template<typename T>
void NDTensor<T>::buildTensor(){
    if(dims.empty()) {
        std::cout << "cant initialize empty tensor" << std::endl;
    }else {
        for(int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        for(int i = 0; i < rank; i++) {
            for(int j = 0; j < size; j++) {
                values.push_back(1);
            }
        }
    }
}

template<typename T>
T* NDTensor<T>::data() {
    return values.data();
}