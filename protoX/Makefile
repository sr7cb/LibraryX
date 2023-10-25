PROTO_PATH=/Users/anant/Desktop/workspace/liftingcpp/proto
CONSTANTS= -DDIM=2 -DHDF5=off -DMPI=off
CFLAGS=-g

proto:
	G++ -std=c++20 -I$(PROTO_PATH)/include -I. -Wunused-result -O3 $(CONSTANTS)  testleveleuler.cpp 
