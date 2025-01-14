FFTW_HOME=/home/sanilr/fftw-3.3.10

all: test

test:
	g++-9 -std=c++17 -O3 -w hockney_test.cpp -o cpu_hockney -lfftw3

cuda:
	nvcc -O3 -w hockney_test_cuda.cu -o cuda_hockney -lcufft

hip:
	hipcc -O3 -w hockney_test_hip.cpp -o hip_hockney -lrocfft

mkl:
	icpx -ggdb -fsycl hockney_test_mkl.cpp -o mkl_hockney -lmkl_sycl -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lOpenCL	

hproto:
	g++-9 -std=c++17 -w -DDIM=3 -O3 -I../ProtoX/proto/include hockney_proto.cpp -lfftw3

clean:
	rm *_hockney