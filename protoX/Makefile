PROTO_PATH=/home/sanilr/proto
CONSTANTS= -DDIM=2 -DHDF5=off -DMPI=off
HIPCONSTANTS = -DHIP=on
CUDACONSTANTS = -DCUDA=on
CFLAGS=-g

proto:
	g++ -std=c++17 -I$(PROTO_PATH)/include -I. -Wunused-result -O3 $(CONSTANTS)  testleveleuler.cpp 

proto_cuda:
	nvcc -std=c++17 -I$(PROTO_PATH)/include -I. -O3 $(CONSTANTS) $(CUDACONSTANTS) testleveleuler.cpp 

proto_hip:
	hipcc -std=c++17 -I$(PROTO_PATH)/include -I. -Wunused-result -O3 $(CONSTANTS) testleveleuler.cpp 