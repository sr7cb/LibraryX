FFTW_HOME=/home/sanilr/fftw-3.3.10/

all: test

test:
	g++-9 -ggdb -I$(FFTW_HOME)/include test_psatd.cpp -o cpu_psatd -lfftw3

cuda:
	nvcc -ccbin g++-9 -std=c++17 -g -G test_psatd_cuda.cu -o cuda_psatd -lcufft