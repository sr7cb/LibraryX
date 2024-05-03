CXX = g++
CXXFLAGS = -g -std=c++20
NEWCXXFLAGS =  -std=c++20 -Wall -Wfatal-errors -Wconversion -fsanitize=undefined,address -DDEBUG -D_GLIBCXX_DEBUG -g
LDLIBS = -lm -lfftw3 -ldl

convolve: main.cpp
	$(CXX) $(CXXFLAGS) -DHP main.cpp -o convolve $(LDLIBS)

clean:
	rm -rf *.o *.out convolve
