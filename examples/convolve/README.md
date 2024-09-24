## Shim example

### Compile and Run

If the files are not including the std namespace.
```bash
g++ -DHP -lm -lfftw3 main.cpp
```

If the files are including the std namespace

```bash
g++ -DHP_USING_STD -lm -lfftw3 main.cpp
```

### Approach discussion

We define the methods from fftw which we want to shim inside a custom namespace `spiral_fftw`.
These methods are part of a shim.hpp header file. These methods log a message and then call back the original method. So they effectively intercepts the fftw method calls.

Next, we try to override the existing fftw methods by using the following two approaches

#### Approach 1

Here, we add our `shim.hpp` at the top of the file, remove the fftw3.h include because it is included in `shim.hpp`. And add the following `using namespace spiral_fftw;` at the top of the file. The expectation was that the fftw methods will first be searched in the `spiral_fftw` namespace and hence we would be able to intercept the method.
However, this is not allowed in C++ as the two methods are now in conflict and compiler does not know which one to call and when.


#### Approach 2

Here, instead of adding the methods of `spiral_fftw` namespace, we use `#if defined` statements to prefix our custom namespace `spiral_fftw` before the methods to be intercepted.

Example 1: 
When std namespace is not included with the using directive. (recommended)
```bash
convolve.hpp and shim.hpp
g++ -DHP -lm -lfftw3 main.cpp
```


Example 2: 
When std namespace is included with the using directive. 
```bash
convolve2.hpp and shim.hpp
g++ -DHP_USING_STD -lm -lfftw3 main.cpp
```
