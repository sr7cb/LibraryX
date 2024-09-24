all: library test

library:
	g++ -fPIC -shared cpp_create.cpp -o liblibraryX.so

test:
	gcc -I. test_c_interface.c -L. -llibraryX -lstdc++ -Wl,-rpath=.