PROTO_PATH=/home/sanilr/proto
CONSTANTS= -DDIM=2 -DHDF5=off -DMPI=off
CFLAGS=-g

proto:
	g++ -I$(PROTO_PATH)/include -I. -Wunused-result -O3 $(CONSTANTS)  testleveleuler.cpp 