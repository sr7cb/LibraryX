#include <iostream>
#include <vector>

__global__ void extractPencil(double* pencil, double *values, int loc, int stride, int rank) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < rank) {
        pencil[tid] = values[tid*stride + loc];
        pencil[tid+1] = values[tid*stride + loc+1];
    }
}

__global__ void setPencil(double* values, double* pencil, int loc, int stride, int rank) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < rank) {
        values[tid*stride + loc] = pencil[tid];
        values[tid*stride + loc+1] = pencil[tid+1];
    }
}

// __global__ void buildValues(double* values, int size) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid == 0)
//         printf("hello from tid 0 with size %d\n", size);
//     if(tid < size)
//         values[tid] = 1;
// }

__global__ void buildValues(double** values, int rank, int* sizes, int totalsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0)
        printf("hello from tid 0 with size %d\n", totalsize);

    if(tid < totalsize){
        int lrank = -1;
        int running_sum = 0;
        for(int i = 0; i < rank; i++){
            running_sum += sizes[i];
            if(tid < running_sum) {
                lrank = i;
            }
        }
        values[lrank][tid%sizes[lrank]] = 1;
    }
}


template<typename T>
class NDTensor{
    private:
    int rank;
    int size;
    std::vector<int> dims;
    int stride;
    T** values;
    T** larray;

    public:

        NDTensor() {
            rank = 1;
            dims.push_back(1);
            size = 1;
            // values = new T*[rank];
            
            // for(int i = 0; i < rank; i++) {
            //     // values[i] = new T[size];
            //     cudaMalloc(&values[i], size*sizeof(T));
            // }
            cudaMalloc(&values, rank*size*sizeof(T*));
            T* larray;
            for(int i = 0; i < rank; i++){
                cudaMalloc(&larray, size*sizeof(T));
            }
            cudaMemcpy(values, larray, rank*sizeof(T*), cudaMemcpyHostToDevice);
        }

        NDTensor(int r, std::vector<std::vector<int>> in_dims = {{1},{2},{3}}) {
            rank = r;
            // values = new T*[rank];
            cudaMalloc(&values, rank*sizeof(T*));
            larray = (T**)malloc(rank*sizeof(T*));
            std::cout << in_dims.size() << std::endl;
            size = -1;
            stride = in_dims.at(0).size();
            
            for(int i = 0; i < in_dims.size(); i++) {
                int lsize = 1;
                for(int j = 0; j < in_dims.at(i).size(); j++){
                    std::cout << in_dims.at(i).size() << "\t";
                    std::cout << in_dims.at(i).at(j) << "\t";
                    dims.push_back(in_dims.at(i).at(j));
                    std::cout << dims.back() << std::endl;
                    lsize = lsize * in_dims.at(i).at(j);
                    
                }
                std::cout << lsize << std::endl;
                cudaMalloc(&larray[i], lsize*sizeof(T));  
               std::cout << std::endl;
            }
            // std::cout << "Malloc with size " << lsize << std::endl;
            cudaMemcpy(values, larray, rank*sizeof(T*), cudaMemcpyHostToDevice);
            
            
        }

        NDTensor(int r, std::vector<int> in_dims = {1,2,3}) {
            rank = r;
            // // values = new T*[rank];
            cudaMalloc(&values, rank*sizeof(T*));
            larray = (T**)malloc(rank*sizeof(T*));
            dims = in_dims;
            stride = in_dims.size();
            size = 1;
            for(int i = 0; i < in_dims.size(); i++) {
                size *= in_dims[i];
                cudaMalloc(&larray[i], size*sizeof(T));  
            }
            cudaMemcpy(values, larray, rank*sizeof(T*), cudaMemcpyHostToDevice);
        }
        // void fillRandom() {
        //     for(int i = 0; i < rank; i++) {
        //         if(size != -1) {
        //             for(int j = 0; j < size; j++) {
        //                 values[i][j] = rand() % 100;
        //             }
        //         } else {
        //             for(int j = 0; j < dims.at(i*stride) * dims.at(i*stride+1) * dims.at(i*stride +2); j++) {
        //                 values[i][j] = rand() % 100;
        //             }
        //         }
        //     }
        // }
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
    // int offset = 0;
    // if(size == -1)
    // {
    //     for(int i = 0; i < n; i++) {
    //         offset += dims.at(i+stride) *dims.at(i*stride+1) *dims.at(i*stride+2);
    //     }
    // } else {
    //     for(int i = 0; i < n; i++) {
    //         offset += dims.at(0) *dims.at(1) *dims.at(2);
    //     }
    // }
    return values[n];

}
template<typename T>
T* NDTensor<T>::getPencil(int i, int j, int k) {
    if(dims.size() > 3){
        std::cout << "error getPencil doesnt work for irregularly shaped tensors" << std::endl;
        exit(-1);
    }
    // std::vector<T> pencil;
    double *pencil;
    cudaMalloc(&pencil, rank*2*sizeof(T));
    if(i*j*k > size){
        std::cout << "point is greater than size" << std::endl;
    } else{
        // for(int l = 0; l < rank; l++) {
        //     pencil.push_back(values[l][i*dims.at(0)*dims.at(1) +j *dims.at(1) + k]);
        // }
        int stride = 1;
        for(int l = 0; l < dims.size(); l++)
            stride *= dims.at(l);
        int loc = i*dims.at(0)*dims.at(1) +j *dims.at(1) + k;
        extractPencil<<<1, rank>>>(pencil, values, loc, stride, rank);
        cudaDeviceSynchronize();

    }
    return pencil;
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
        // for(int l = 0; l < rank; l++) {
        //     values[l][i*dims.at(0)*dims.at(1) +j *dims.at(1) + k] = pencil[l];
        // }
        int stride = 1;
        for(int l = 0; l < dims.size(); l++)
            stride *= dims.at(l);
        int loc = i*dims.at(0)*dims.at(1) +j *dims.at(1) + k;
        setPencil<<<1,rank>>>(values, pencil, loc, stride, rank);
        cudaDeviceSynchronize();
    }
}

template<typename T>
void NDTensor<T>::buildTensor(){
    if(dims.empty()) {
        std::cout << "cant initialize empty tensor" << std::endl;
    }else if(size == -1) {
        std::cout << "irregular tensor" << std::endl;
        // for(int i = 0; i < rank; i++) {
        //     for(int j = 0; j < dims.at(i*stride) * dims.at(i*stride+1)*dims.at(i*stride+2); j++) {
        //         values[i][j] = 1;
        //     }
        // }
        int *lsizes;
        cudaMallocManaged(&lsizes, rank*sizeof(int));
        for(int i = 0; i < rank; i++)
            lsizes[i] = dims.at(i*stride) * dims.at(i*stride+1) * dims.at(i*stride+2);
        int blockSize = 256;
        int totalsize = 1;
        for(int i = 0; i < rank; i++)
            totalsize += lsizes[i];
        int gridSize = (totalsize + blockSize - 1) / blockSize;
        std::cout << gridSize << " " << blockSize  << " " << totalsize << std::endl;
        buildValues<<<gridSize, blockSize>>>(values, rank, lsizes, totalsize);
        cudaDeviceSynchronize();
        cudaFree(lsizes);
    }else {
        // for(int i = 0; i < dims.size(); i++) {
        //     size *= dims[i];
        // }
        // for(int i = 0; i < rank; i++) {
        //     for(int j = 0; j < size; j++) {
        //         values[i][j] = 1;
        //     }
        // }
        int *lsizes;
        cudaMallocManaged(&lsizes, rank*sizeof(int));
        int blockSize = 256;
        int totalsize = 1;
        for(int i = 0; i < dims.size(); i++) {
            totalsize *= dims.size();
        }
        for(int i = 0; i < rank; i++)
            lsizes[i] = totalsize;
        int gridSize = (rank*totalsize + blockSize - 1) / blockSize;
        std::cout << gridSize << " " << blockSize  << " " << totalsize << std::endl;
        buildValues<<<gridSize, blockSize>>>(values, rank, lsizes, totalsize);
        cudaDeviceSynchronize();
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