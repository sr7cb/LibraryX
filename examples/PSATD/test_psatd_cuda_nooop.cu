#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include <cufft.h>
#include <cuComplex.h>
#include <vector> 
#include "spiral_generated_psatd_cuda.hpp"


template <typename T>
void printCuda(T* input, int size, std::string s) {
  T* cpu_out = new T[size];
  cudaMemcpy(cpu_out, input, size*sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << "New obj print " << s << std::endl;
  for(int i = 0; i < 10; i++)
    std::cout << cpu_out[i] << " ";
  std::cout << std::endl;
  std::cout << "New obj print end " << s <<  std::endl;
  free(cpu_out);
}


__global__ void print2DArray(double** input, int fields) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < fields) {
    printf("the first value of each field is %lf\n", input[tid][0]);
  }
}

__global__ void print1DArray(double** input, int index, int tsize) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < tsize) {
    printf("the first value of each field is %lf\n", input[index][tid]);
  }
}

__global__ void initArray(double *input, int size) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < size)
    input[tid] = 1;
}

__global__ void copyArray(double **output, double *input, int size, int index) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < size) {
    output[index][tid] = input[tid];
  }
}

__global__ void copyArray(double *output, double **input, int size, int index) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < size) {
    output[tid] = input[index][tid];
  }
}

int product3(std::vector<int> sizes) {
    int result = 1;
    for(int i = 0; i < sizes.size(); i++)
      result *= sizes.at(i);

    return result;
}

__global__ void extractKernel(int index, double **d_input, double *d_output, 
                              int x, int y, int z, 
                              int x_small, int y_small, int z_small) {
    // Calculate thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if within bounds of the smaller cube
    if (i < x_small && j < y_small && k < z_small) {
        // Calculate the index in the larger input vector
        int inputIndex = i * y * z + j * z + k;
        
        // Calculate the index in the smaller output vector
        int outputIndex = i * y_small * z_small + j * z_small + k;

        // Copy the value from input to output
        d_output[outputIndex] = d_input[index][inputIndex];
    }
}

__device__ cuDoubleComplex calculate_w(int n, int k) {
    const double PI = 3.14159265358979323846;
    double angle = 2.0 * PI * k / n;
    return make_cuDoubleComplex(cos(angle), sin(angle));
}

//shift each points based on shifts vector
__global__ void apply_shift(cuDoubleComplex* fft_out, int3 dims, int3 shifts, int total_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;
    // Compute the 3D indices from the 1D index
    int i = index / (dims.x * dims.y);
    int j = (index / dims.y) % dims.x;
    int k = index % dims.y;

    cuDoubleComplex shift = make_cuDoubleComplex(1.0, 0.0);

    if (shifts.x != 0) {
        shift = cuCmul(shift, calculate_w(dims.x, j));
    }
    if (shifts.y != 0) {
        shift = cuCmul(shift, calculate_w(dims.y, k));
    }
    if (shifts.z != 0) {
        shift = cuCmul(shift, calculate_w(dims.z, i));
    }

    // Apply the shift to the corresponding fft_out element
    fft_out[index] = cuCmul(shift, fft_out[index]);
}

void Resample(const std::vector<int>& out = {1, 2, 3}, 
                const std::vector<int>& in = {1, 2, 3},  
                const std::vector<double>& shifts = {1, 2, 3}, 
                double **output = NULL, 
                double **input = NULL,
                int index = 0,
                cufftHandle plan = -1,
                cufftHandle plan2 = -1) {
    // print1DArray<<<1,10>>>(input, 0);
    // cudaDeviceSynchronize();

    double *d_output;
    cudaMalloc(&d_output, out.at(0) * out.at(1) * out.at(2) * sizeof(double));

    // Define block dimensions for 256 threads per block
    dim3 blockDim(8, 8, 4);

    // Calculate grid dimensions
    dim3 gridDim((out.at(0) + blockDim.x - 1) / blockDim.x,
                 (out.at(1) + blockDim.y - 1) / blockDim.y,
                 (out.at(2) + blockDim.z - 1) / blockDim.z);

    // Launch the kernel with the updated configuration
    extractKernel<<<gridDim, blockDim>>>(index, input, d_output, in.at(0), in.at(1), in.at(2), out.at(0), out.at(1), out.at(2));

    double* fft_out;
    cudaMalloc(&fft_out, out.at(0)*out.at(1)*((out.at(2)/2)+1) *2 *sizeof(double));
    // cufftHandle plan;
    // cufftPlan3d(&plan, out.at(0), out.at(1), out.at(2), CUFFT_D2Z);
    cufftExecD2Z(plan, (cufftDoubleReal *)d_output, (cufftDoubleComplex *)fft_out);

    
    int3 dims = make_int3(out.at(0), out.at(1), ((out.at(2)/2)+1));
    int3 dshifts = make_int3(shifts[0], shifts[1], shifts[2]);
    int total_elements = dims.x * dims.y * dims.z;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    apply_shift<<<blocksPerGrid, threadsPerBlock>>>((cuDoubleComplex*)fft_out, dims, dshifts, total_elements);

    double * loutput;
    cudaMalloc(&loutput, out.at(0)*out.at(1)*out.at(2)*sizeof(double));
    // cufftHandle plan2;
    // cufftPlan3d(&plan2, out.at(0), out.at(1), out.at(2), CUFFT_Z2D);
    cufftExecZ2D(plan2, (cufftDoubleComplex *)fft_out, (cufftDoubleReal *)loutput);
    
    int elems = (out.at(0)*out.at(1)*out.at(2) + threadsPerBlock -1)/(threadsPerBlock);
    copyArray<<<elems, threadsPerBlock>>>(output, loutput, out.at(0)*out.at(1)*out.at(2), index);
    cudaFree(d_output);
    cudaFree(fft_out);
    cudaFree(loutput);
}

//  __forceinline__ __device__ void complexMultiplyKernel( double resReal, double resImag, double realA, double imagA, double realB, double imagB) {

//   resReal += realA * realB - imagA * imagB;  // Real part of the result
//   resImag += realA * imagB + imagA * realB; // Imaginary part of the result

// }


__global__ void createvals(double *values, double **sym, int length, int i, int j, int k, int dimx, int dimy, int dimz, int n, double c2, double invep0) {
  int tid = blockIdx.x *blockDim.x +threadIdx.x;

  if(tid < length) {
    double fmkx = sym[0][j];
    double fmky = sym[1][k];
    double fmkz = sym[2][i];
    double fcv = sym[3][i*dimy*dimx + j *dimy + k];
    double fsckv = sym[4][i*dimy*dimx + j *dimy + k];
    double fx1v = sym[5][i*dimy*dimx + j *dimy + k];
    double fx2v = sym[6][i*dimy*dimx + j *dimy + k];
    double fx3v = sym[7][i*dimy*dimx + j *dimy + k];

    if(tid%2 == 0) {
      values[tid] = 0;
    } else if(tid== 1) {
      values[threadIdx.x] = fcv / (n*n*n);
    } else if(tid== 3) {
      values[tid] = (-fmkz * c2 * fsckv) / (n*n*n);
    }else if(tid== 5) {
      values[tid] = (fmkz * c2 * fsckv )/ (n*n*n);
    }else if(tid== 7) {
      values[tid] = (-invep0 * fsckv) / (n*n*n);
    }else if(tid== 9) {
      values[tid] = (fmkx * fx3v) / (n*n*n);
    }else if(tid== 11) { //end first row
      values[tid] = (-fmkx * fx2v) / (n*n*n);
    }else if(tid== 13) {
      values[tid] = fcv / (n*n*n);
    }else if(tid== 15) {
      values[tid] = (-fmkz * c2 * fsckv) / (n*n*n);
    }else if(tid== 17) {
      values[tid] = (fmkx * c2 * fsckv )/ (n*n*n);
    }else if(tid== 19) {
      values[tid] = (-invep0 * fsckv) / (n*n*n);
    }else if(tid== 21) {
      values[tid] = fmky * fx3v / (n*n*n);
    }else if(tid== 23) {//end second row
      values[tid] = -fmky * fx2v / (n*n*n);
    }else if(tid== 25) {
      values[tid] = fcv / (n*n*n);
    }else if(tid== 27) {
      values[tid] = (-fmky * c2 * fsckv) / (n*n*n);
    }else if(tid== 29) {
      values[tid] = (fmkx * c2 * fsckv )/ (n*n*n);
    }else if(tid== 31) {
      values[tid] = (-invep0 * fsckv) / (n*n*n);
    }else if(tid== 33) {
      values[tid] = (fmkz * fx3v) / (n*n*n);
    }else if(tid== 35) { //end third row
      values[tid] = (-fmkz * fx2v) / (n*n*n);
    }else if(tid== 37) {
      values[tid] = (fmkz * fsckv) / (n*n*n);
    }else if(tid== 39) {
      values[tid] = (-fmky * fsckv )/ (n*n*n);
    }else if(tid== 41) {
      values[tid] = (fcv) / (n*n*n);
    } else if(tid== 43) {
      values[tid] = -fmkz * fx1v / (n*n*n);
    } else if(tid== 45) { //end fourth row
      values[tid] = fmky * fx1v / (n*n*n);
    } else if(tid== 47) {
      values[tid] = (fmkz * fsckv) / (n*n*n);
    } else if(tid== 49) {
      values[tid] = (-fmkx * fsckv )/ (n*n*n);
    } else if(tid== 51) {
      values[tid] = (fcv) / (n*n*n);
    } else if(tid== 53) {
      values[tid] = (fmkz * fx1v) / (n*n*n);
    } else if(tid== 55) { //end fifth row
      values[tid] = -fmkx * fx1v / (n*n*n);
    } else if(tid== 57) {
      values[tid] = (fmky * fsckv) / (n*n*n);
    } else if(tid== 59) {
      values[tid] = (-fmkx * fsckv )/ (n*n*n);
    } else if(tid== 61) {
      values[tid] = (fcv) / (n*n*n);
    } else if(tid== 63) {
      values[tid] = (-fmky * fx1v) / (n*n*n);
    }else if(tid== 65) {
      values[tid] = (fmkx * fx1v) / (n*n*n);
    }    
  }
}

__global__ void spmv(double **output, double **input, int index, int length, int *rows, int *cols, double *vals) {
  int tid = blockIdx.x *blockDim.x +threadIdx.x;
  for(int i = tid; i < length; i+=blockDim.x*gridDim.x) {
    double outputreal = 0;
    double outputcom = 0;
    for(int j = rows[i]; j < rows[i+1]; j++) {
      outputreal += input[cols[j]][2*index] * vals[2*j] - input[cols[j]][2*index+1] * vals[2*j+1];  
      outputcom += input[cols[j]][2*index] * vals[2*j+1] + input[cols[j]][2*index+1] * vals[2*j];
    }
    output[i][2*index] = outputreal;
    output[i][2*index+1] = outputcom;
  }
}

struct WarpXconfig{
    int n = 80;
    int np = n+1;
    int inFields = 11;
    int outFields = 6;
    int nf = n+2;
    int xdim = nf/2;
    int ydim = n;
    int zdim = n;
    double cvar = 10;
    double ep0var = 10;
    double c2 = cvar*cvar;
    double invep0 = 1/ep0var; 
    double** symbol;
};


int main() {

  WarpXconfig conf;
    int n = conf.n;
    int np = conf.np;
    int nf = conf.nf;
    int symFields = 8;
    std::vector<std::vector<int> > input_sizes = {{np, np, n},
                                           {np, n, np},
                                           {n, np, np},
                                           {n, n, np},
                                           {n, np, n},
                                           {np, n, n},
                                           {np, np, n},
                                           {np, n, np},
                                           {n, np, np},
                                           {np, np, np},
                                           {np, np, np}};  
    
    std::vector<std::vector<int> > output_sizes = {{np, np, n},
                                             {np, n, np},
                                             {n, np, np},
                                             {n, n, np},
                                             {n, np, n},
                                             {np, n, n}};  
                                           
  double **cudain, **hostin;                                         
  cudaMalloc     ( &cudain, sizeof(double) * conf.inFields );
	cudaMallocHost ( &hostin, sizeof(double) * conf.inFields );
	for (int comp = 0; comp < conf.inFields; comp++) {
		cudaMalloc ( &hostin[comp], sizeof(double) * product3(input_sizes[comp]) );
    int blockSize = 256;
    int gridSize = (product3(input_sizes[comp]) + blockSize -1)/(blockSize);
    initArray<<<gridSize, blockSize>>>(hostin[comp], product3(input_sizes[comp]));
	}
	cudaMemcpy ( cudain, hostin, sizeof(double) * conf.inFields, cudaMemcpyHostToDevice );

  double **cudasym, **hostsym;                                         
  cudaMalloc     ( &cudasym, sizeof(double) * symFields );
	cudaMallocHost ( &hostsym, sizeof(double) * symFields );
	for (int comp = 0; comp < symFields; comp++) {
    if(comp == 0) {
		  cudaMalloc ( &hostsym[comp], sizeof(double) * nf/2 );
      int blockSize = 256;
      int gridSize = ((nf/2)+blockSize-1)/blockSize;
      initArray<<<gridSize, blockSize>>>(hostsym[comp], (nf/2));
    } else if(comp == 1) {
      cudaMalloc ( &hostsym[comp], sizeof(double) * n );
		  int blockSize = 256;
      int gridSize = (n + blockSize -1)/blockSize;
      initArray<<<gridSize, blockSize>>>(hostsym[comp], n);
    } else if (comp == 2) {
      cudaMalloc ( &hostsym[comp], sizeof(double) * n );
		  int blockSize = 256;
      int gridSize = (n + blockSize - 1)/blockSize;
      initArray<<<gridSize, blockSize>>>(hostsym[comp], n);
    } else {
      cudaMalloc ( &hostsym[comp], sizeof(double) * (nf/2)* n *n);
      int blockSize = 256;
      int gridSize = (((nf/2)* n *n) + blockSize -1)/blockSize;
      initArray<<<gridSize, blockSize>>>(hostsym[comp], ((nf/2)* n *n));
    }
	}
	cudaMemcpy ( cudasym, hostsym, sizeof(double) * symFields, cudaMemcpyHostToDevice );

  double **cudaout, **hostout;                                         
  cudaMalloc     ( &cudaout, sizeof(double) * conf.outFields );
	cudaMallocHost ( &hostout, sizeof(double) * conf.outFields );
	for (int comp = 0; comp < conf.outFields; comp++) {
		cudaMalloc ( &hostout[comp], sizeof(double) * product3(output_sizes[comp]) );
    int blockSize = 256;
    int gridSize = (product3(output_sizes[comp]) + blockSize -1)/blockSize;
    initArray<<<gridSize, blockSize>>>(hostout[comp], product3(output_sizes[comp]));
	}
	cudaMemcpy ( cudaout, hostout, sizeof(double) * conf.outFields, cudaMemcpyHostToDevice );

  double **boxBig0, **hostbb0;
  cudaMalloc     ( &boxBig0, sizeof(double) * conf.inFields );
	cudaMallocHost ( &hostbb0, sizeof(double) * conf.inFields );

  for (int comp = 0; comp < conf.inFields; comp++) {
		cudaMalloc ( &hostbb0[comp], sizeof(double) * n*n*n );
		int blockSize = 256;
    int gridSize = ((n*n*n)+ blockSize - 1)/blockSize;
    initArray<<<gridSize, blockSize>>>(hostbb0[comp], (n*n*n));
	}
	cudaMemcpy ( boxBig0, hostbb0, sizeof(double) * conf.inFields, cudaMemcpyHostToDevice );

  double **boxBig1, **hostbb1;
  cudaMalloc     ( &boxBig1, sizeof(double) * conf.inFields );
	cudaMallocHost ( &hostbb1, sizeof(double) * conf.inFields );

  for (int comp = 0; comp < conf.inFields; comp++) {
		cudaMalloc ( &hostbb1[comp], sizeof(double) * n*n*nf*2 );
		int blockSize = 256;
    int gridSize = ((n*n*nf*2) + blockSize -1)/blockSize;
    initArray<<<gridSize, blockSize>>>(hostbb1[comp], (n*n*nf*2));
	}
	cudaMemcpy ( boxBig1, hostbb1, sizeof(double) * conf.inFields, cudaMemcpyHostToDevice );

  double **boxBig2, **hostbb2;
  cudaMalloc     ( &boxBig2, sizeof(double) * conf.outFields );
	cudaMallocHost ( &hostbb2, sizeof(double) * conf.outFields );

  for (int comp = 0; comp < conf.outFields; comp++) {
		cudaMalloc ( &hostbb2[comp], sizeof(double) * n*n*nf*2 );
		int blockSize = 256;
    int gridSize = ((n*n*nf*2) + blockSize - 1)/blockSize;
    initArray<<<gridSize, blockSize>>>(hostbb2[comp], (n*n*nf*2));
	}
	cudaMemcpy ( boxBig2, hostbb2, sizeof(double) * conf.outFields, cudaMemcpyHostToDevice );

  double **boxBig3, **hostbb3;
  cudaMalloc     ( &boxBig3, sizeof(double) * conf.outFields );
	cudaMallocHost ( &hostbb3, sizeof(double) * conf.outFields );

  for (int comp = 0; comp < conf.outFields; comp++) {
		cudaMalloc ( &hostbb3[comp], sizeof(double) * n*n*n );
		int blockSize = 256;
    int gridSize = ((n*n*n) + blockSize - 1)/blockSize;
    initArray<<<gridSize, blockSize>>>(hostbb3[comp], (n*n*n));
	}
	cudaMemcpy ( boxBig3, hostbb3, sizeof(double) * conf.outFields, cudaMemcpyHostToDevice );


  /*print2DArray<<<1,conf.inFields>>>(cudain, conf.inFields);
  cudaDeviceSynchronize();
  std::cout << std::endl;
  print2DArray<<<1,symFields>>>(cudasym, symFields);
  cudaDeviceSynchronize();
  std::cout << std::endl;
  print2DArray<<<1,conf.outFields>>>(cudaout, conf.inFields);
  cudaDeviceSynchronize();
  std::cout << std::endl; 
  print2DArray<<<1,conf.inFields>>>(boxBig0, conf.inFields);
  cudaDeviceSynchronize(); 
  std::cout << std::endl;
  print2DArray<<<1,conf.inFields>>>(boxBig1, conf.inFields);
  cudaDeviceSynchronize(); 
  std::cout << std::endl;
  print2DArray<<<1,conf.inFields>>>(boxBig2, conf.outFields);
  cudaDeviceSynchronize(); 
  std::cout << std::endl;
  print2DArray<<<1,conf.inFields>>>(boxBig3, conf.outFields);
  cudaDeviceSynchronize(); 
  std::cout << std::endl;*/

  cufftHandle resample_forward_plan;
  cufftPlan3d(&resample_forward_plan, n,n,n, CUFFT_D2Z);
  cufftHandle resample_inverse_plan;
  cufftPlan3d(&resample_inverse_plan, n,n,n, CUFFT_Z2D);

  double * finput;
  cudaMalloc(&finput, n*n*n*sizeof(double));
  double * foutput;
  cudaMalloc(&foutput, n*n*(n/2+1)*2*sizeof(double));
  int blockSize = 256;
  int ingridSize = ((n*n*n)+blockSize-1)/blockSize;
  int outgridSize = ((n*n*(n/2+1)*2) + blockSize-1)/(blockSize);
  cufftHandle plan1;
  cufftPlan3d(&plan1, n, n, n, CUFFT_D2Z);

  std::vector<int> rows{0,6,12,18,23,28,33};
  std::vector<int> cols{0,4,5,6,9,10,
                            1,3,5,7,9,10,
                            2,3,4,8,9,10,
                            1,2,3,7,8,
                            0,2,4,6,8,
                            0,1,5,6,7};

  int *drows, *dcols;
  cudaMalloc(&drows, rows.size()*sizeof(int));
  cudaMalloc(&dcols, cols.size()*sizeof(int));
  cudaMemcpy(drows, rows.data(), rows.size()*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dcols, cols.data(), cols.size()*sizeof(int), cudaMemcpyHostToDevice);
  double *cvals;
  cudaMalloc(&cvals, cols.size()*2*sizeof(double));

  double * ifinput;
  cudaMalloc(&ifinput, n*n*(n/2+1)*2*sizeof(double));
  double * ifoutput;
  cudaMalloc(&ifoutput, n*n*n*sizeof(double));
  int iingridSize = ((n*n*(n/2+1)*2) + blockSize-1)/(blockSize);
  int ioutgridSize = ((n*n*n)+blockSize-1)/blockSize;
  cufftHandle plan2;
  cufftPlan3d(&plan2, n, n, n, CUFFT_Z2D);

    cufftHandle forwardplan1;
  cufftPlan3d(&forwardplan1, np, np, n, CUFFT_D2Z);
  cufftHandle forwardplan2;
  cufftPlan3d(&forwardplan2, np, n, np, CUFFT_D2Z);
  cufftHandle forwardplan3;
  cufftPlan3d(&forwardplan3, n, np, np, CUFFT_D2Z);
  cufftHandle forwardplan4;
  cufftPlan3d(&forwardplan4, n, n, np, CUFFT_D2Z);
  cufftHandle forwardplan5;
  cufftPlan3d(&forwardplan5, n, np, n, CUFFT_D2Z);
  cufftHandle forwardplan6;
  cufftPlan3d(&forwardplan6, np, n, n, CUFFT_D2Z);

  cufftHandle inverseplan1;
  cufftPlan3d(&inverseplan1, np, np, n, CUFFT_Z2D);
  cufftHandle inverseplan2;
  cufftPlan3d(&inverseplan2, np, n, np, CUFFT_Z2D);
  cufftHandle inverseplan3;
  cufftPlan3d(&inverseplan3, n, np, np, CUFFT_Z2D);
  cufftHandle inverseplan4;
  cufftPlan3d(&inverseplan4, n, n, np, CUFFT_Z2D);
  cufftHandle inverseplan5;
  cufftPlan3d(&inverseplan5, n, np, n, CUFFT_Z2D);
  cufftHandle inverseplan6;
  cufftPlan3d(&inverseplan6, np, n, n, CUFFT_Z2D);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  Resample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0, cudain, 0, resample_forward_plan, resample_inverse_plan);
  Resample({n, n, n}, {np, np, n}, {0.0, -0.5, 0.0}, boxBig0, cudain, 1, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 1), nth(X, 1),
  Resample({n, n, n}, {np, np, np}, {-0.5, 0.0, 0.0}, boxBig0, cudain, 2, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 2), nth(X, 2),
  Resample({n, n, n}, {n, n, np}, {-0.5, -0.5, 0.0}, boxBig0, cudain, 3, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]), nth(boxBig0, 3), nth(X, 3),
  Resample({n, n, n}, {n, np, n}, {-0.5, 0.0, -0.5}, boxBig0, cudain, 4, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]), nth(boxBig0, 4), nth(X, 4),
  Resample({n, n, n}, {np, n, n}, {0.0, -0.5, -0.5}, boxBig0, cudain, 5, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]), nth(boxBig0, 5), nth(X, 5),
  Resample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0, cudain, 6, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 6), nth(X, 6),
  Resample({n, n, n}, {np, n, np}, {0.0, -0.5, 0.0}, boxBig0, cudain, 7, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 7), nth(X, 7),
  Resample({n, n, n}, {n, np, np}, {-0.5, 0.0, 0.0}, boxBig0, cudain, 8, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 8), nth(X, 8),
  Resample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0, cudain, 9, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 9), nth(X, 9),
  Resample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0, cudain, 10, resample_forward_plan, resample_inverse_plan);// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 10), nth(X, 10),

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 0);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 0);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 1);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 1);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 2);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 2);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 3);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 3);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 4);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 4);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 5);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 5);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 6);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 6);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 7);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 7);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 8);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 8);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 9);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 9);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 10);
  cufftExecD2Z(plan1, (cufftDoubleReal *)finput, (cufftDoubleComplex *)foutput);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 10);

  
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < nf; j++) {
      for(int k = 0; k < n; k++) {
        createvals<<<1,128>>>(cvals, cudasym, 2*cols.size(), i,j,k, nf/2,n,n,n,conf.c2,conf.invep0);
        spmv<<<1,128>>>(boxBig2, boxBig1, i*(n*nf) + j*n + k, rows.size()-1, drows, dcols, cvals);
      }
    }
  } 

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 0);
  cufftExecZ2D(plan2, (cufftDoubleComplex *)ifinput, (cufftDoubleReal *)ifoutput);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 0);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 1);
  cufftExecZ2D(plan2, (cufftDoubleComplex *)ifinput, (cufftDoubleReal *)ifoutput);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 1);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 2);
  cufftExecZ2D(plan2, (cufftDoubleComplex *)ifinput, (cufftDoubleReal *)ifoutput);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 2);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 3);
  cufftExecZ2D(plan2, (cufftDoubleComplex *)ifinput, (cufftDoubleReal *)ifoutput);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 3);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 4);
  cufftExecZ2D(plan2, (cufftDoubleComplex *)ifinput, (cufftDoubleReal *)ifoutput);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 4);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 5);
  cufftExecZ2D(plan2, (cufftDoubleComplex *)ifinput, (cufftDoubleReal *)ifoutput);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 5);
  
  Resample({np, np, n}, {n, n, n}, {0.0, 0.0, 0.5}, cudaout, boxBig3,0, forwardplan1, inverseplan1);// TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]), nth(Y, 0), nth(boxBig3, 0),
  Resample({np, n, np}, {n, n, n}, {0.0, 0.5, 0.0}, cudaout, boxBig3,1, forwardplan2, inverseplan2);// TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]), nth(Y, 1), nth(boxBig3, 1),
  Resample({n, np, np}, {n, n, n}, {0.5, 0.0, 0.0}, cudaout, boxBig3,2, forwardplan3, inverseplan3);// TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]), nth(Y, 2), nth(boxBig3, 2),

  Resample({n, n, np}, {n, n, n}, {0.5, 0.5, 0.0}, cudaout, boxBig3,3, forwardplan4, inverseplan4);// TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]), nth(Y, 3), nth(boxBig3, 3),
  Resample({n, np, n}, {n, n, n}, {0.5, 0.0, 0.5}, cudaout, boxBig3,4, forwardplan5, inverseplan5);// TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]), nth(Y, 4), nth(boxBig3, 4),
  Resample({np, n, n}, {n, n, n}, {0.0, 0.5, 0.5}, cudaout, boxBig3,5, forwardplan6, inverseplan6);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Time taken is " << milliseconds << " ms" << std::endl;

  init_warpx_cuda();
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2);
  
  warpx_cuda(cudaout, cudain, cudasym, conf.cvar, conf.ep0var);
  
  cudaEventRecord(stop2);
  cudaEventSynchronize(stop2);
  float milliseconds2 = 0;
  cudaEventElapsedTime(&milliseconds2, start2, stop2);
  std::cout << "SPIRAL time taken: " << milliseconds2 << std::endl;
  destroy_warpx_cuda();

  return 0;
}

/*__global__ void super_spmv_kernel(double **output, double **input, double **sym, int *rows, int *cols, 
                        int row_size, const int col_size, int n, int dimx, int dimy, int dimz, double c2, double invep0) {
  // int tid = blockIdx.x *blockDim.x + threadIdx.x;
  
  __shared__ double values[2*33];//colsize
  for(int index = blockIdx.x; index < dimx*dimy*(dimz*2); index+=gridDim.x) {
    int x = index/(dimy*dimz);
    int y = (index % (dimy*dimz))/dimz;
    int z = (index % dimz);

    double fmkx = sym[0][y];
    double fmky = sym[1][z];
    double fmkz = sym[2][x];
    // double fcv = sym[3][index];
    double fsckv = sym[4][index];
    double fx1v = sym[5][index];
    double fx2v = sym[6][index];
    double fx3v = sym[7][index];

    if(threadIdx.x %2 == 0 && threadIdx.x < (2*33)) {
      values[threadIdx.x] = 0;
    } else if(threadIdx.x == 1) {
      // values[threadIdx.x] = fcv / (n*n*n);
      values[threadIdx.x] = sym[3][index] / (n*n*n);
    } else if(threadIdx.x == 3) {
      values[threadIdx.x] = (-fmkz * c2 * fsckv) / (n*n*n);
    }else if(threadIdx.x == 5) {
      values[threadIdx.x] = (fmkz * c2 * fsckv )/ (n*n*n);
    }else if(threadIdx.x == 7) {
      values[threadIdx.x] = (-invep0 * fsckv) / (n*n*n);
    }else if(threadIdx.x == 9) {
      values[threadIdx.x] = (fmkx * fx3v) / (n*n*n);
    }else if(threadIdx.x == 11) { //end first row
      values[threadIdx.x] = (-fmkx * fx2v) / (n*n*n);
    }else if(threadIdx.x == 13) {
      // values[threadIdx.x] = fcv / (n*n*n);
      values[threadIdx.x] = sym[3][index] / (n*n*n);
    }else if(threadIdx.x == 15) {
      values[threadIdx.x] = (-fmkz * c2 * fsckv) / (n*n*n);
    }else if(threadIdx.x == 17) {
      values[threadIdx.x] = (fmkx * c2 * fsckv )/ (n*n*n);
    }else if(threadIdx.x == 19) {
      values[threadIdx.x] = (-invep0 * fsckv) / (n*n*n);
    }else if(threadIdx.x == 21) {
      values[threadIdx.x] = fmky * fx3v / (n*n*n);
    }else if(threadIdx.x == 23) {//end second row
      values[threadIdx.x] = -fmky * fx2v / (n*n*n);
    }else if(threadIdx.x == 25) {
      // values[threadIdx.x] = fcv / (n*n*n);
      values[threadIdx.x] = sym[3][index] / (n*n*n);
    }else if(threadIdx.x == 27) {
      values[threadIdx.x] = (-fmky * c2 * fsckv) / (n*n*n);
    }else if(threadIdx.x == 29) {
      values[threadIdx.x] = (fmkx * c2 * fsckv )/ (n*n*n);
    }else if(threadIdx.x == 31) {
      values[threadIdx.x] = (-invep0 * fsckv) / (n*n*n);
    }else if(threadIdx.x == 33) {
      values[threadIdx.x] = (fmkz * fx3v) / (n*n*n);
    }else if(threadIdx.x == 35) { //end third row
      values[threadIdx.x] = (-fmkz * fx2v) / (n*n*n);
    }else if(threadIdx.x == 37) {
      values[threadIdx.x] = (fmkz * fsckv) / (n*n*n);
    }else if(threadIdx.x == 39) {
      values[threadIdx.x] = (-fmky * fsckv )/ (n*n*n);
    }else if(threadIdx.x == 41) {
      // values[threadIdx.x] = (fcv) / (n*n*n);
      values[threadIdx.x] = sym[3][index] / (n*n*n);
    } else if(threadIdx.x == 43) {
      values[threadIdx.x] = -fmkz * fx1v / (n*n*n);
    } else if(threadIdx.x == 45) { //end fourth row
      values[threadIdx.x] = fmky * fx1v / (n*n*n);
    } else if(threadIdx.x == 47) {
      values[threadIdx.x] = (fmkz * fsckv) / (n*n*n);
    } else if(threadIdx.x == 49) {
      values[threadIdx.x] = (-fmkx * fsckv )/ (n*n*n);
    } else if(threadIdx.x == 51) {
      // values[threadIdx.x] = (fcv) / (n*n*n);
      values[threadIdx.x] = sym[3][index] / (n*n*n);
    } else if(threadIdx.x == 53) {
      values[threadIdx.x] = (fmkz * fx1v) / (n*n*n);
    } else if(threadIdx.x == 55) { //end fifth row
      values[threadIdx.x] = -fmkx * fx1v / (n*n*n);
    } else if(threadIdx.x == 57) {
      values[threadIdx.x] = (fmky * fsckv) / (n*n*n);
    } else if(threadIdx.x == 59) {
      values[threadIdx.x] = (-fmkx * fsckv )/ (n*n*n);
    } else if(threadIdx.x == 61) {
      // values[threadIdx.x] = (fcv) / (n*n*n);
      values[threadIdx.x] = sym[3][index] / (n*n*n);
    } else if(threadIdx.x == 63) {
      values[threadIdx.x] = (-fmky * fx1v) / (n*n*n);
    }else if(threadIdx.x == 65) {
      values[threadIdx.x] = (fmkx * fx1v) / (n*n*n);
    }    
    __syncthreads();

    for(int i = threadIdx.x; i < row_size; i+=blockDim.x) {
      double outreal = 0;
      double outimag = 0;
      for(int j = rows[i]; j < rows[i+1]; j++) {
        complexMultiplyKernel(outreal, outimag, values[j], values[j+1], input[cols[j]][index*2], input[cols[j]][index*2+1]);
      }
      output[i][index*2] = outreal;
      output[i][index*2+1] = outimag;
    }
    
  }
}*/

// std::vector<int> rows{0,6,12,18,23,28,33};
  // std::vector<int> cols{0,4,5,6,9,10,
  //                           1,3,5,7,9,10,
  //                           2,3,4,8,9,10,
  //                           1,2,3,7,8,
  //                           0,2,4,6,8,
  //                           0,1,5,6,7};

  // int *drows, *dcols;
  // cudaMalloc(&drows, rows.size()*sizeof(int));
  // cudaMalloc(&dcols, cols.size()*sizeof(int));
  // cudaMemcpy(drows, rows.data(), rows.size()*sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(dcols, cols.data(), cols.size()*sizeof(int), cudaMemcpyHostToDevice);
  // int spmvblock = 128;
  // int spmvgrid = (n*n*nf + spmvblock -1)/spmvblock;
  // super_spmv_kernel<<<spmvgrid, spmvblock>>>(boxBig2, boxBig1, cudasym, drows, dcols, 
  //                       rows.size(), cols.size(), n, n, n, nf/2, conf.c2, conf.invep0);
  // cudaDeviceSynchronize();  
  // print2DArray<<<1,conf.outFields>>>(boxBig2, conf.outFields);
  // cudaDeviceSynchronize();