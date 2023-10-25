#include <functional>
#include "/Users/anant/Desktop/workspace/liftingcpp/proto/include/Proto.H"
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

namespace spiral_fftw {
    template <class T=double, unsigned int C=1,
    Operator::MemType MEM=MEMTYPE_DEFAULT,
    unsigned char D=1, unsigned char E=1>
    class BoxData : public Proto::BoxData<T, C, MEM, D, E>{
        using Proto::BoxData<T, C, MEM, D, E>;

        BoxData& operator+=(const BoxData& rhs){
            std::cout << "We are inside operator+=" << std::endl;

            this += rhs;
            Proto::BoxData<T, C, MEM, D, E>::operator+=(rhs);
            return *this;
        }
    };
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
#define BoxData spiral_fftw::BoxData