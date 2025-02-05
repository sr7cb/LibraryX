#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector> 
#include <rocfft/rocfft.h>
#include "spiral_generated_psatd_hip.hpp"


template <typename T>
void printCuda(T* input, int size, std::string s) {
  T* cpu_out = new T[size];
  hipMemcpy(cpu_out, input, size*sizeof(T), hipMemcpyDeviceToHost);
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

__device__ hipDoubleComplex calculate_w(int n, int k) {
    const double PI = 3.14159265358979323846;
    double angle = 2.0 * PI * k / n;
    return make_hipDoubleComplex(cos(angle), sin(angle));
}

//shift each points based on shifts vector
__global__ void apply_shift(hipDoubleComplex* fft_out, int3 dims, int3 shifts, int total_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;
    // Compute the 3D indices from the 1D index
    int i = index / (dims.x * dims.y);
    int j = (index / dims.y) % dims.x;
    int k = index % dims.y;

    hipDoubleComplex shift = make_hipDoubleComplex(1.0, 0.0);

    if (shifts.x != 0) {
        shift = hipCmul(shift, calculate_w(dims.x, j));
    }
    if (shifts.y != 0) {
        shift = hipCmul(shift, calculate_w(dims.y, k));
    }
    if (shifts.z != 0) {
        shift = hipCmul(shift, calculate_w(dims.z, i));
    }

    // Apply the shift to the corresponding fft_out element
    fft_out[index] = hipCmul(shift, fft_out[index]);
}

void Resample(const std::vector<int>& out = {1, 2, 3}, 
                const std::vector<int>& in = {1, 2, 3},  
                const std::vector<double>& shifts = {1, 2, 3}, 
                double **output = NULL, 
                double **input = NULL,
                int index = 0,
                rocfft_plan forward_plan = NULL,
                rocfft_plan inverse_plan = NULL,
                rocfft_execution_info forward_info = NULL,
                rocfft_execution_info inverse_info = NULL) {
    // print1DArray<<<1,10>>>(input, 0);
    // cudaDeviceSynchronize();

    double *d_output;
    hipMalloc(&d_output, out.at(0) * out.at(1) * out.at(2) * sizeof(double));

    // Define block dimensions for 256 threads per block
    dim3 blockDim(8, 8, 4);

    // Calculate grid dimensions
    dim3 gridDim((out.at(0) + blockDim.x - 1) / blockDim.x,
                 (out.at(1) + blockDim.y - 1) / blockDim.y,
                 (out.at(2) + blockDim.z - 1) / blockDim.z);

    // Launch the kernel with the updated configuration
    // extractKernel<<<gridDim, blockDim>>>(index, input, d_output, in.at(0), in.at(1), in.at(2), out.at(0), out.at(1), out.at(2));
    hipLaunchKernelGGL(extractKernel, gridDim, blockDim, 0, 0, index, input, d_output, in.at(0), in.at(1), in.at(2), out.at(0), out.at(1), out.at(2));

    double* fft_out;
    hipMalloc(&fft_out, out.at(0)*out.at(1)*((out.at(2)/2)+1) *2 *sizeof(double));
    // rocfft_plan forward_plan;
    // size_t work_size = 0;
    // size_t lengths[3] = {static_cast<size_t>(out.at(0)), static_cast<size_t>(out.at(1)), static_cast<size_t>(out.at(2))};
    // rocfft_plan_create(&forward_plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
    //                    3, lengths, 1, nullptr);

    // rocfft_plan_get_work_buffer_size(forward_plan, &work_size);
    // void* work_buffer;
    // hipMalloc(&work_buffer, work_size);
    // rocfft_execution_info forward_info;
    // rocfft_execution_info_create(&forward_info);
    // rocfft_execution_info_set_work_buffer(forward_info, work_buffer, work_size);

    rocfft_execute(forward_plan, (void**)&d_output, (void**)&fft_out, forward_info);

    
    int3 dims = make_int3(out.at(0), out.at(1), ((out.at(2)/2)+1));
    int3 dshifts = make_int3(shifts[0], shifts[1], shifts[2]);
    int total_elements = dims.x * dims.y * dims.z;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    apply_shift<<<blocksPerGrid, threadsPerBlock>>>((hipDoubleComplex*)fft_out, dims, dshifts, total_elements);
    hipLaunchKernelGGL(apply_shift, blocksPerGrid, threadsPerBlock, 0, 0, (hipDoubleComplex*)fft_out, dims, dshifts, total_elements);

    double * loutput;
    hipMalloc(&loutput, out.at(0)*out.at(1)*out.at(2)*sizeof(double));
    // rocfft_plan inverse_plan;
    // rocfft_plan_create(&inverse_plan, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
    //                    3, lengths, 1, nullptr);

    // rocfft_execution_info inverse_info;
    // rocfft_execution_info_create(&inverse_info);
    // rocfft_execution_info_set_work_buffer(inverse_info, work_buffer, work_size);

    rocfft_execute(inverse_plan, (void**)&fft_out, (void**)&loutput, inverse_info);
    
    int elems = (out.at(0)*out.at(1)*out.at(2) + threadsPerBlock -1)/(threadsPerBlock);
    // copyArray<<<elems, threadsPerBlock>>>(output, loutput, out.at(0)*out.at(1)*out.at(2), index);
    hipLaunchKernelGGL(copyArray, elems, threadsPerBlock, 0, 0, output, loutput, out.at(0)*out.at(1)*out.at(2), index);
    hipFree(d_output);
    hipFree(fft_out);
    hipFree(loutput);
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

__global__ void threadspmv(double **output, double **input, double **sym, int totalelem, int length, int *rows, int *cols, int dimx, int dimy, int dimz, int n, int c2, int invep0) {
  int tid = blockIdx.x *blockDim.x +threadIdx.x;
  if(tid < totalelem) {
    int i = tid / ((dimx*2) * dimy);
    int j = (tid / dimy) % (dimx*2);
    int k = tid % dimy;
    double fmkx = sym[0][j];
    double fmky = sym[1][k];
    double fmkz = sym[2][i];
    double fcv = sym[3][i*dimy*dimx + j *dimy + k];
    double fsckv = sym[4][i*dimy*dimx + j *dimy + k];
    double fx1v = sym[5][i*dimy*dimx + j *dimy + k];
    double fx2v = sym[6][i*dimy*dimx + j *dimy + k];
    double fx3v = sym[7][i*dimy*dimx + j *dimy + k];
    double values[66];
    for(int i = 0; i < 66; i++) {
      if(i%2 == 0) {
        values[i] = 0;
      } else if(i == 1) {
      values[i] = fcv / (n*n*n);
      } else if(i == 3) {
        values[i] = (-fmkz * c2 * fsckv) / (n*n*n);
      }else if(i == 5) {
        values[i] = (fmkz * c2 * fsckv )/ (n*n*n);
      }else if(i == 7) {
        values[i] = (-invep0 * fsckv) / (n*n*n);
      }else if(i == 9) {
        values[i] = (fmkx * fx3v) / (n*n*n);
      }else if(i== 11) { //end first row
        values[i] = (-fmkx * fx2v) / (n*n*n);
      }else if(i== 13) {
        values[i] = fcv / (n*n*n);
      }else if(i== 15) {
        values[i] = (-fmkz * c2 * fsckv) / (n*n*n);
      }else if(i== 17) {
        values[i] = (fmkx * c2 * fsckv )/ (n*n*n);
      }else if(i== 19) {
        values[i] = (-invep0 * fsckv) / (n*n*n);
      }else if(i== 21) {
        values[i] = fmky * fx3v / (n*n*n);
      }else if(i== 23) {//end second row
        values[i] = -fmky * fx2v / (n*n*n);
      }else if(i== 25) {
        values[i] = fcv / (n*n*n);
      }else if(i== 27) {
        values[i] = (-fmky * c2 * fsckv) / (n*n*n);
      }else if(i== 29) {
        values[i] = (fmkx * c2 * fsckv )/ (n*n*n);
      }else if(i== 31) {
        values[i] = (-invep0 * fsckv) / (n*n*n);
      }else if(i== 33) {
        values[i] = (fmkz * fx3v) / (n*n*n);
      }else if(i== 35) { //end third row
        values[i] = (-fmkz * fx2v) / (n*n*n);
      }else if(i== 37) {
        values[i] = (fmkz * fsckv) / (n*n*n);
      }else if(i== 39) {
        values[i] = (-fmky * fsckv )/ (n*n*n);
      }else if(i== 41) {
        values[i] = (fcv) / (n*n*n);
      } else if( i == 43) {
        values[i] = -fmkz * fx1v / (n*n*n);
      } else if(i== 45) { //end fourth row
        values[i] = fmky * fx1v / (n*n*n);
      } else if(i== 47) {
        values[i] = (fmkz * fsckv) / (n*n*n);
      } else if(i== 49) {
        values[i] = (-fmkx * fsckv )/ (n*n*n);
      } else if(i== 51) {
        values[i] = (fcv) / (n*n*n);
      } else if(i== 53) {
        values[i] = (fmkz * fx1v) / (n*n*n);
      } else if(i== 55) { //end fifth row
        values[i] = -fmkx * fx1v / (n*n*n);
      } else if(i== 57) {
        values[i] = (fmky * fsckv) / (n*n*n);
      } else if(i== 59) {
        values[i] = (-fmkx * fsckv )/ (n*n*n);
      } else if(i== 61) {
        values[i] = (fcv) / (n*n*n);
      } else if(i== 63) {
        values[i] = (-fmky * fx1v) / (n*n*n);
      }else if(i== 65) {
        values[i] = (fmkx * fx1v) / (n*n*n);
      } 
    }
    for(int i = 0; i < length; i++) {
      double outputreal = 0;
      double outputcom = 0;
      for(int j = rows[i]; j < rows[i+1]; j++) {
        outputreal += input[cols[j]][2*tid] * values[2*j] - input[cols[j]][2*tid+1] * values[2*j+1];  
        outputcom += input[cols[j]][2*tid] * values[2*j+1] + input[cols[j]][2*tid+1] * values[2*j];
      }
      output[i][2*tid] = outputreal;
      output[i][2*tid+1] = outputcom;
    }
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
  hipMalloc     ( &cudain, sizeof(double) * conf.inFields );
	hipHostMalloc ( &hostin, sizeof(double) * conf.inFields, hipHostMallocDefault );
	for (int comp = 0; comp < conf.inFields; comp++) {
		hipMalloc ( &hostin[comp], sizeof(double) * product3(input_sizes[comp]) );
    int blockSize = 256;
    int gridSize = (product3(input_sizes[comp]) + blockSize -1)/(blockSize);
    // initArray<<<gridSize, blockSize>>>(hostin[comp], product3(input_sizes[comp]));
    hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostin[comp], product3(input_sizes[comp]));
	}
	hipMemcpy ( cudain, hostin, sizeof(double) * conf.inFields, hipMemcpyHostToDevice );
 
  double **cudasym, **hostsym;                                         
  hipMalloc     ( &cudasym, sizeof(double) * symFields );
	hipHostMalloc ( &hostsym, sizeof(double) * symFields, hipHostMallocDefault );
	for (int comp = 0; comp < symFields; comp++) {
    if(comp == 0) {
		  hipMalloc ( &hostsym[comp], sizeof(double) * nf/2 );
      int blockSize = 256;
      int gridSize = ((nf/2)+blockSize-1)/blockSize;
      // initArray<<<gridSize, blockSize>>>(hostsym[comp], (nf/2));
      hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostsym[comp], (nf/2));
    } else if(comp == 1) {
      hipMalloc ( &hostsym[comp], sizeof(double) * n );
		  int blockSize = 256;
      int gridSize = (n + blockSize -1)/blockSize;
      // initArray<<<gridSize, blockSize>>>(hostsym[comp], n);
      hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostsym[comp], n);
    } else if (comp == 2) {
      hipMalloc ( &hostsym[comp], sizeof(double) * n );
		  int blockSize = 256;
      int gridSize = (n + blockSize - 1)/blockSize;
      // initArray<<<gridSize, blockSize>>>(hostsym[comp], n);
      hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostsym[comp], n);
    } else {
      hipMalloc ( &hostsym[comp], sizeof(double) * (nf/2)* n *n);
      int blockSize = 256;
      int gridSize = (((nf/2)* n *n) + blockSize -1)/blockSize;
      // initArray<<<gridSize, blockSize>>>(hostsym[comp], ((nf/2)* n *n));
      hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostsym[comp], ((nf/2)* n *n));
    }
	}
	hipMemcpy ( cudasym, hostsym, sizeof(double) * symFields, hipMemcpyHostToDevice );

  double **cudaout, **hostout;                                         
  hipMalloc     ( &cudaout, sizeof(double) * conf.outFields );
	hipHostMalloc ( &hostout, sizeof(double) * conf.outFields, hipHostMallocDefault );
	for (int comp = 0; comp < conf.outFields; comp++) {
		hipMalloc ( &hostout[comp], sizeof(double) * product3(output_sizes[comp]) );
    int blockSize = 256;
    int gridSize = (product3(output_sizes[comp]) + blockSize -1)/blockSize;
    // initArray<<<gridSize, blockSize>>>(hostout[comp], product3(output_sizes[comp]));
    hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostout[comp], product3(output_sizes[comp]));
	}
	hipMemcpy ( cudaout, hostout, sizeof(double) * conf.outFields, hipMemcpyHostToDevice );

  double **boxBig0, **hostbb0;
  hipMalloc     ( &boxBig0, sizeof(double) * conf.inFields );
	hipHostMalloc ( &hostbb0, sizeof(double) * conf.inFields, hipHostMallocDefault );

  for (int comp = 0; comp < conf.inFields; comp++) {
		hipMalloc ( &hostbb0[comp], sizeof(double) * n*n*n );
		int blockSize = 256;
    int gridSize = ((n*n*n)+ blockSize - 1)/blockSize;
    // initArray<<<gridSize, blockSize>>>(hostbb0[comp], (n*n*n));
    hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostbb0[comp], (n*n*n));
	}
	hipMemcpy ( boxBig0, hostbb0, sizeof(double) * conf.inFields, hipMemcpyHostToDevice );

  double **boxBig1, **hostbb1;
  hipMalloc     ( &boxBig1, sizeof(double) * conf.inFields );
	hipHostMalloc ( &hostbb1, sizeof(double) * conf.inFields, hipHostMallocDefault );

  for (int comp = 0; comp < conf.inFields; comp++) {
		hipMalloc ( &hostbb1[comp], sizeof(double) * n*n*nf*2 );
		int blockSize = 256;
    int gridSize = ((n*n*nf*2) + blockSize -1)/blockSize;
    // initArray<<<gridSize, blockSize>>>(hostbb1[comp], (n*n*nf*2));
    hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostbb1[comp], (n*n*nf*2));
	}
	hipMemcpy ( boxBig1, hostbb1, sizeof(double) * conf.inFields, hipMemcpyHostToDevice );

  double **boxBig2, **hostbb2;
  hipMalloc     ( &boxBig2, sizeof(double) * conf.outFields );
	hipHostMalloc ( &hostbb2, sizeof(double) * conf.outFields, hipHostMallocDefault );

  for (int comp = 0; comp < conf.outFields; comp++) {
		hipMalloc ( &hostbb2[comp], sizeof(double) * n*n*nf*2 );
		int blockSize = 256;
    int gridSize = ((n*n*nf*2) + blockSize - 1)/blockSize;
    // initArray<<<gridSize, blockSize>>>(hostbb2[comp], (n*n*nf*2));
    hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostbb2[comp], (n*n*nf*2));
	}
	hipMemcpy ( boxBig2, hostbb2, sizeof(double) * conf.outFields, hipMemcpyHostToDevice );

  double **boxBig3, **hostbb3;
  hipMalloc     ( &boxBig3, sizeof(double) * conf.outFields );
	hipHostMalloc ( &hostbb3, sizeof(double) * conf.outFields, hipHostMallocDefault );

  for (int comp = 0; comp < conf.outFields; comp++) {
		hipMalloc ( &hostbb3[comp], sizeof(double) * n*n*n );
		int blockSize = 256;
    int gridSize = ((n*n*n) + blockSize - 1)/blockSize;
    // initArray<<<gridSize, blockSize>>>(hostbb3[comp], (n*n*n));
    hipLaunchKernelGGL(initArray, gridSize, blockSize, 0, 0, hostbb3[comp], (n*n*n));
	}
	hipMemcpy ( boxBig3, hostbb3, sizeof(double) * conf.outFields, hipMemcpyHostToDevice );


  // print2DArray<<<1,conf.inFields>>>(cudain, conf.inFields);
  // hipDeviceSynchronize();
  // std::cout << std::endl;
  // print2DArray<<<1,symFields>>>(cudasym, symFields);
  // hipDeviceSynchronize();
  // std::cout << std::endl;
  // print2DArray<<<1,conf.outFields>>>(cudaout, conf.inFields);
  // hipDeviceSynchronize();
  // std::cout << std::endl; 
  // print2DArray<<<1,conf.inFields>>>(boxBig0, conf.inFields);
  // hipDeviceSynchronize(); 
  // std::cout << std::endl;
  // print2DArray<<<1,conf.inFields>>>(boxBig1, conf.inFields);
  // hipDeviceSynchronize(); 
  // std::cout << std::endl;
  // print2DArray<<<1,conf.inFields>>>(boxBig2, conf.outFields);
  // hipDeviceSynchronize(); 
  // std::cout << std::endl;
  // print2DArray<<<1,conf.inFields>>>(boxBig3, conf.outFields);
  // hipDeviceSynchronize(); 
  // std::cout << std::endl;

 /*initial resample plans*/
  rocfft_plan resample_forward_plan;
  size_t work_size = 0;
  size_t resample_lengths[3] = {static_cast<size_t>(n), static_cast<size_t>(n), static_cast<size_t>(n)};
  rocfft_plan_create(&resample_forward_plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, resample_lengths, 1, nullptr);

  rocfft_plan_get_work_buffer_size(resample_forward_plan, &work_size);
  void* work_buffer;
  hipMalloc(&work_buffer, work_size);
  rocfft_execution_info resample_forward_info;
  rocfft_execution_info_create(&resample_forward_info);
  rocfft_execution_info_set_work_buffer(resample_forward_info, work_buffer, work_size);

  rocfft_plan resample_inverse_plan;
  rocfft_plan_create(&resample_inverse_plan, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, resample_lengths, 1, nullptr);

  rocfft_execution_info resample_inverse_info;
  rocfft_execution_info_create(&resample_inverse_info);
  rocfft_execution_info_set_work_buffer(resample_inverse_info, work_buffer, work_size);
  /*initial resample plans*/

  /*fft plans*/
   rocfft_plan forward_plan;
  size_t lengths[3] = {static_cast<size_t>(n), static_cast<size_t>(n), static_cast<size_t>(n)};
  rocfft_plan_create(&forward_plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths, 1, nullptr);

  rocfft_plan_get_work_buffer_size(forward_plan, &work_size);
  void* work_buffer2;
  hipMalloc(&work_buffer2, work_size);
  rocfft_execution_info forward_info;
  rocfft_execution_info_create(&forward_info);
  rocfft_execution_info_set_work_buffer(forward_info, work_buffer2, work_size);

  rocfft_plan inverse_plan;
  rocfft_plan_create(&inverse_plan, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths, 1, nullptr);

  rocfft_execution_info inverse_info;
  rocfft_execution_info_create(&inverse_info);
  rocfft_execution_info_set_work_buffer(inverse_info, work_buffer, work_size);
  /*fft plans*/

  /*output resample plans*/
  rocfft_plan output_forward_plan1;
  size_t lengths1[3] = {static_cast<size_t>(np), static_cast<size_t>(np), static_cast<size_t>(n)};
  rocfft_plan_create(&output_forward_plan1, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths1, 1, nullptr);
  rocfft_plan_get_work_buffer_size(output_forward_plan1, &work_size);
  rocfft_execution_info forward_info1;
  rocfft_execution_info_create(&forward_info1);
  rocfft_execution_info_set_work_buffer(forward_info1, work_buffer, work_size);
  rocfft_plan output_forward_plan2;
  size_t lengths2[3] = {static_cast<size_t>(np), static_cast<size_t>(n), static_cast<size_t>(np)};
  rocfft_plan_create(&output_forward_plan2, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths2, 1, nullptr);
  rocfft_plan_get_work_buffer_size(output_forward_plan2, &work_size);
  rocfft_execution_info forward_info2;
  rocfft_execution_info_create(&forward_info2);
  rocfft_execution_info_set_work_buffer(forward_info2, work_buffer, work_size);
  rocfft_plan output_forward_plan3;
  size_t lengths3[3] = {static_cast<size_t>(n), static_cast<size_t>(np), static_cast<size_t>(np)};
  rocfft_plan_create(&output_forward_plan3, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths3, 1, nullptr);
  rocfft_plan_get_work_buffer_size(output_forward_plan3, &work_size);
  rocfft_execution_info forward_info3;
  rocfft_execution_info_create(&forward_info3);
  rocfft_execution_info_set_work_buffer(forward_info3, work_buffer, work_size);
  rocfft_plan output_forward_plan4;
  size_t lengths4[3] = {static_cast<size_t>(n), static_cast<size_t>(n), static_cast<size_t>(np)};
  rocfft_plan_create(&output_forward_plan4, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths4, 1, nullptr);
  rocfft_plan_get_work_buffer_size(output_forward_plan4, &work_size);
  rocfft_execution_info forward_info4;
  rocfft_execution_info_create(&forward_info4);
  rocfft_execution_info_set_work_buffer(forward_info4, work_buffer, work_size);
  rocfft_plan output_forward_plan5;
  size_t lengths5[3] = {static_cast<size_t>(n), static_cast<size_t>(np), static_cast<size_t>(n)};
  rocfft_plan_create(&output_forward_plan5, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths5, 1, nullptr);
  rocfft_plan_get_work_buffer_size(output_forward_plan5, &work_size);
  rocfft_execution_info forward_info5;
  rocfft_execution_info_create(&forward_info5);
  rocfft_execution_info_set_work_buffer(forward_info5, work_buffer, work_size);
  rocfft_plan output_forward_plan6;
  size_t lengths6[3] = {static_cast<size_t>(np), static_cast<size_t>(n), static_cast<size_t>(n)};
  rocfft_plan_create(&output_forward_plan6, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                      3, lengths6, 1, nullptr);
  rocfft_plan_get_work_buffer_size(output_forward_plan6, &work_size);
  rocfft_execution_info forward_info6;
  rocfft_execution_info_create(&forward_info6);
  rocfft_execution_info_set_work_buffer(forward_info6, work_buffer, work_size);

  rocfft_plan output_inverse_plan1;
  rocfft_plan_create(&output_inverse_plan1, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths1, 1, nullptr);
  rocfft_execution_info inverse_info1;
  rocfft_execution_info_create(&inverse_info1);
  rocfft_execution_info_set_work_buffer(inverse_info1, work_buffer, work_size);

  rocfft_plan output_inverse_plan2;
  rocfft_plan_create(&output_inverse_plan2, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths2, 1, nullptr);
  rocfft_execution_info inverse_info2;
  rocfft_execution_info_create(&inverse_info2);
  rocfft_execution_info_set_work_buffer(inverse_info2, work_buffer, work_size);

  rocfft_plan output_inverse_plan3;
  rocfft_plan_create(&output_inverse_plan3, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths3, 1, nullptr);
  rocfft_execution_info inverse_info3;
  rocfft_execution_info_create(&inverse_info3);
  rocfft_execution_info_set_work_buffer(inverse_info3, work_buffer, work_size);

  rocfft_plan output_inverse_plan4;
  rocfft_plan_create(&output_inverse_plan4, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths4, 1, nullptr);
  rocfft_execution_info inverse_info4;
  rocfft_execution_info_create(&inverse_info4);
  rocfft_execution_info_set_work_buffer(inverse_info4, work_buffer, work_size);

  rocfft_plan output_inverse_plan5;
  rocfft_plan_create(&output_inverse_plan5, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths5, 1, nullptr);
  rocfft_execution_info inverse_info5;
  rocfft_execution_info_create(&inverse_info5);
  rocfft_execution_info_set_work_buffer(inverse_info5, work_buffer, work_size);

  rocfft_plan output_inverse_plan6;
  rocfft_plan_create(&output_inverse_plan6, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                      3, lengths6, 1, nullptr);
  rocfft_execution_info inverse_info6;
  rocfft_execution_info_create(&inverse_info6);
  rocfft_execution_info_set_work_buffer(inverse_info6, work_buffer, work_size);
  /*output resample plans*/

  /*memory creation*/
  double * finput;
  hipMalloc(&finput, n*n*n*sizeof(double));
  double * foutput;
  hipMalloc(&foutput, n*n*(n/2+1)*2*sizeof(double));
  int blockSize = 256;
  int ingridSize = ((n*n*n)+blockSize-1)/blockSize;
  int outgridSize = ((n*n*(n/2+1)*2) + blockSize-1)/(blockSize);

  std::vector<int> rows{0,6,12,18,23,28,33};
  std::vector<int> cols{0,4,5,6,9,10,
                            1,3,5,7,9,10,
                            2,3,4,8,9,10,
                            1,2,3,7,8,
                            0,2,4,6,8,
                            0,1,5,6,7};

  int *drows, *dcols;
  hipMalloc(&drows, rows.size()*sizeof(int));
  hipMalloc(&dcols, cols.size()*sizeof(int));
  hipMemcpy(drows, rows.data(), rows.size()*sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(dcols, cols.data(), cols.size()*sizeof(int), hipMemcpyHostToDevice);
  double *cvals;
  hipMalloc(&cvals, cols.size()*2*sizeof(double));

  double * ifinput;
  hipMalloc(&ifinput, n*n*(n/2+1)*2*sizeof(double));
  double * ifoutput;
  hipMalloc(&ifoutput, n*n*n*sizeof(double));
  int iingridSize = ((n*n*(n/2+1)*2) + blockSize-1)/(blockSize);
  int ioutgridSize = ((n*n*n)+blockSize-1)/blockSize;
  /*memory creation*/

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start);
  Resample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0, cudain, 0, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);
  Resample({n, n, n}, {np, np, n}, {0.0, -0.5, 0.0}, boxBig0, cudain, 1, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 1), nth(X, 1),
  Resample({n, n, n}, {np, np, np}, {-0.5, 0.0, 0.0}, boxBig0, cudain, 2, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 2), nth(X, 2),
  Resample({n, n, n}, {n, n, np}, {-0.5, -0.5, 0.0}, boxBig0, cudain, 3, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]), nth(boxBig0, 3), nth(X, 3),
  Resample({n, n, n}, {n, np, n}, {-0.5, 0.0, -0.5}, boxBig0, cudain, 4, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]), nth(boxBig0, 4), nth(X, 4),
  Resample({n, n, n}, {np, n, n}, {0.0, -0.5, -0.5}, boxBig0, cudain, 5, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]), nth(boxBig0, 5), nth(X, 5),
  Resample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0, cudain, 6, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 6), nth(X, 6),
  Resample({n, n, n}, {np, n, np}, {0.0, -0.5, 0.0}, boxBig0, cudain, 7, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 7), nth(X, 7),
  Resample({n, n, n}, {n, np, np}, {-0.5, 0.0, 0.0}, boxBig0, cudain, 8, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 8), nth(X, 8),
  Resample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0, cudain, 9, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 9), nth(X, 9),
  Resample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0, cudain, 10, resample_forward_plan, resample_inverse_plan, resample_forward_info, resample_inverse_info);// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 10), nth(X, 10),
 

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 0);
  hipLaunchKernelGGL(copyArray, ingridSize, blockSize, 0, 0, finput, boxBig1, n*n*n, 0);
   rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 0);
  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 1);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 1);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 2);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 2);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 3);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 3);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 4);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 4);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 5);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 5);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 6);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 6);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 7);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 7);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 8);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 8);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 9);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 9);

  copyArray<<<ingridSize, blockSize>>>(finput, boxBig1, n*n*n, 10);
  rocfft_execute(forward_plan, (void**)&finput, (void**)&foutput, forward_info);
  copyArray<<<outgridSize, blockSize>>>(boxBig1, foutput, n*n*(n/2+1)*2, 10);


  // for(int i = 0; i < n; i++) {
  //   for(int j = 0; j < nf; j++) {
  //     for(int k = 0; k < n; k++) {
  //       // createvals<<<1,128>>>(cvals, cudasym, 2*cols.size(), i,j,k, nf/2,n,n,n,conf.c2,conf.invep0);
  //       hipLaunchKernelGGL(createvals, 1, 128, 0, 0, cvals, cudasym, 2*cols.size(), i,j,k, nf/2,n,n,n,conf.c2,conf.invep0);
  //       // spmv<<<1,128>>>(boxBig2, boxBig1, i*(n*nf) + j*n + k, rows.size()-1, drows, dcols, cvals);
  //       hipLaunchKernelGGL(spmv, 1, 128, 0, 0, boxBig2, boxBig1, i*(n*nf) + j*n + k, rows.size()-1, drows, dcols, cvals);
  //     }
  //   }
  // } 

  int grid = ((n*n*nf)+blockSize-1)/blockSize;
  threadspmv<<<grid,blockSize>>>(boxBig2, boxBig1, cudasym, (n*n*nf), rows.size()-1, drows, dcols, nf/2, n, n, n, conf.c2, conf.invep0);
  // cudaDeviceSynchronize();
  // print2DArray<<<1,conf.outFields>>>(boxBig2, conf.outFields);
  
  
  

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 0);
  rocfft_execute(inverse_plan, (void**)&ifinput, (void**)&ifoutput, inverse_info);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 0);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 1);
  rocfft_execute(inverse_plan, (void**)&ifinput, (void**)&ifoutput, inverse_info);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 1);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 2);
  rocfft_execute(inverse_plan, (void**)&ifinput, (void**)&ifoutput, inverse_info);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 2);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 3);
  rocfft_execute(inverse_plan, (void**)&ifinput, (void**)&ifoutput, inverse_info);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 3);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 4);
  rocfft_execute(inverse_plan, (void**)&ifinput, (void**)&ifoutput, inverse_info);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 4);

  copyArray<<<iingridSize, blockSize>>>(ifinput, boxBig2, n*n*(n/2+1)*2, 5);
  rocfft_execute(inverse_plan, (void**)&ifinput, (void**)&ifoutput, inverse_info);
  copyArray<<<ioutgridSize, blockSize>>>(boxBig3, foutput, n*n*n, 5);

  Resample({np, np, n}, {n, n, n}, {0.0, 0.0, 0.5}, cudaout, boxBig3,0, output_forward_plan1, output_inverse_plan1, forward_info1, inverse_info1);// TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]), nth(Y, 0), nth(boxBig3, 0),
  Resample({np, n, np}, {n, n, n}, {0.0, 0.5, 0.0}, cudaout, boxBig3,1, output_forward_plan2, output_inverse_plan2, forward_info2, inverse_info2);// TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]), nth(Y, 1), nth(boxBig3, 1),
  Resample({n, np, np}, {n, n, n}, {0.5, 0.0, 0.0}, cudaout, boxBig3,2, output_forward_plan3, output_inverse_plan3, forward_info3, inverse_info3);// TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]), nth(Y, 2), nth(boxBig3, 2),

  Resample({n, n, np}, {n, n, n}, {0.5, 0.5, 0.0}, cudaout, boxBig3,3, output_forward_plan4, output_inverse_plan4, forward_info4, inverse_info4);// TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]), nth(Y, 3), nth(boxBig3, 3),
  Resample({n, np, n}, {n, n, n}, {0.5, 0.0, 0.5}, cudaout, boxBig3,4, output_forward_plan5, output_inverse_plan5, forward_info5, inverse_info5);// TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]), nth(Y, 4), nth(boxBig3, 4),
  Resample({np, n, n}, {n, n, n}, {0.0, 0.5, 0.5}, cudaout, boxBig3,5, output_forward_plan6, output_inverse_plan6, forward_info6, inverse_info6);
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time taken: " << milliseconds << std::endl;
  
  init_warpx_cuda();
  hipEvent_t start2, stop2;
  hipEventCreate(&start2);
  hipEventCreate(&stop2);
  hipEventRecord(start2);
  
  warpx_cuda(cudaout, cudain, cudasym, conf.cvar, conf.ep0var);
  
  hipEventRecord(stop2);
  hipEventSynchronize(stop2);
  float milliseconds2 = 0;
  hipEventElapsedTime(&milliseconds2, start2, stop2);
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