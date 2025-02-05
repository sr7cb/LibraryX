FFTW_HOME=/home/sanilr/fftw-3.3.10/

all: test

test:
	g++-9 -ggdb -I$(FFTW_HOME)/include test_psatd.cpp -o cpu_psatd -lfftw3

cuda:
	nvcc -ccbin g++-9 -std=c++17 -g -G test_psatd_cuda.cu -o cuda_psatd -lcufft

hip:
	hipcc -w -O3 test_psatd_hip_nooop.cpp -o new_hip_psatd -lrocfft

mkl:
	icpx -g -fsycl test_psatd_sycl.cpp -o mkl_psatd -lmkl_sycl -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lOpenCL

nooop:
	nvcc -ccbin g++-9 -std=c++17 -g -G test_psatd_cuda_nooop.cu -o nooop -lcufft
