#include <iostream>
#include <vector>
class NDTensor{
    private:
    int rank;
    int size;
    std::vector<int> dims;
    std::vector<double> values;

    public:
        NDTensor() {
            rank = 1;
            dims.push_back(1);
        }
        NDTensor(int r, std::vector<int> in_dims) {
            if(r != in_dims.size()) {
                std::cout << "rank of tensor must match number of dimensions for dim sizes" << std::endl;
            } else{
                rank = r;
                dims = dims;
            }
        }
        void buildTensor();
        double * getCube(int n);
        double* data();
};

double* NDTensor::getCube(int n) {
    if(n == 0)
        return values.data();
    else {
        int lsize = 0;
        for(int i = 0; i < n; i++)
            lsize += size;
        return values.data() + lsize;
    }

}

void NDTensor::buildTensor(){
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

double* NDTensor::data() {
    return values.data();
}