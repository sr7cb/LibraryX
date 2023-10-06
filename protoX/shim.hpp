#include <functional>
#include "Proto.H"
#pragma once
namespace Operator {
    #define NUMCOMPS DIM+2
    // template <typename T, typename I>
    // spiral_fftw::forall<>() {
    //     std::cout << "We are inside spiral_fftw::fftw_plan_dft_r2c_1d" << std::endl;
    // }
    namespace spiral_fftw{
        template <typename T>
        Proto::BoxData<T, NUMCOMPS> deconvolve(const BoxData<T, NUMCOMPS>&  a_U) {
            std::cout << "We are inside deconvolve" << std::endl;
            Proto::BoxData<T, NUMCOMPS> dummy(Proto::Box::Cube(1));
            return dummy;
        }
        template <typename T>
        Proto::BoxData<double, NUMCOMPS> _convolve(BoxData<T, NUMCOMPS>&  input2, BoxData<T, NUMCOMPS>&  input1) {
            std::cout << "We are inside convolve" << std::endl;
            Proto::BoxData<T, NUMCOMPS> dummy(Proto::Box::Cube(1));
            return dummy;
        }
    }

}

     template<typename T, unsigned int C=1,
        typename Func, typename... Srcs>
     Proto::BoxData<T, C> spiral_forall(const Func& a_F, Srcs&&... a_srcs){
        std::cout << "We are inside forall" << typeid(a_F).name() << std::endl;
        Proto::BoxData<T, C> dummy(Proto::Box::Cube(1));
        return dummy;
     }

#define forall spiral_forall
#define deconvolve spiral_fftw::deconvolve
#define _convolve spiral_fftw::_convolve