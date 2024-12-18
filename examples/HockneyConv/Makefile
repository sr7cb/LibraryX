FFTW_HOME=/home/sanilr/fftw-3.3.10/

all: test

test:
	g++ -O3 -w -I$(FFTW_HOME)/include hockney_test.cpp -o cpu_hockney -lfftw3

cuda:
	nvcc -O3 -w hockney_test_cuda.cu -o cuda_hockney -lcufft

hip:
	hipcc -O3 -w hockney_test_hip.cpp -o hip_hockney -lrocfft

clean:
	rm *_hockney