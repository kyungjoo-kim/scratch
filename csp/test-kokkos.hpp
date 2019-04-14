#ifndef __TEST_KOKKOS_HPP__
#define __TEST_KOKKOS_HPP__

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Eigendecomposition_Decl.hpp"
#include "KokkosBatched_Eigendecomposition_Serial_Impl.hpp"

/// SpT, HpT, value_type are defined in the main-csp.cpp
namespace TestCSP {

  struct TestKokkos {
    using A_value_type_3d_view = Kokkos::View<value_type***, SpT>;
    using E_value_type_3d_view = Kokkos::View<value_type***, Kokkos::LayoutRight,SpT>;
    using V_value_type_4d_view = Kokkos::View<value_type****,Kokkos::LayoutRight,SpT>;
    using W_value_type_2d_view = Kokkos::View<value_type**,  Kokkos::LayoutRight,SpT>;

    int _N, _Blk;

    A_value_type_3d_view _A;
    E_value_type_3d_view _E;
    V_value_type_4d_view _V;
    W_value_type_2d_view _W;

    int getWorkSpaceSize() {
      return 2*_Blk*_Blk + _Blk*5;
    }

    template<typename ArgViewType>
    void setProblem(const ArgViewType &A) {
      const value_type zero(0);
      Kokkos::deep_copy(_A, A);
      Kokkos::deep_copy(_E, zero);
      Kokkos::deep_copy(_V, zero);
      Kokkos::deep_copy(_W, zero);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int &i) const {
      const int r_val = KokkosBatched::
        SerialEigendecompositionInternal::invoke(_Blk,
                                                 &_A(i,0,0),   int(_A.stride(1)), int(_A.stride(2)),
                                                 &_E(i,0,0),   int(_E.stride(2)),
                                                 &_E(i,1,0),   int(_E.stride(2)),
                                                 &_V(i,0,0,0), int(_V.stride(2)), int(_V.stride(3)),
                                                 &_V(i,1,0,0), int(_V.stride(2)), int(_V.stride(3)),
                                                 &_W(i,0),     int(_W.extent(1)));      
    }

    double runTest() {
      Kokkos::Impl::Timer timer;
      timer.reset();
      {
        const Kokkos::RangePolicy<SpT> policy(0, _N);
        Kokkos::parallel_for(policy, *this);
        Kokkos::fence();
      }
      const double t = timer.seconds();
      return t;
    }

    TestKokkos(const int N, const int Blk)
      : _N(N),
        _Blk(Blk),
        _A("A_kk", N, Blk, Blk),
        _E("E_kk", N, 2, Blk),
        _V("V_kk", N, 2, Blk, Blk),
        _W("W_kk", N, getWorkSpaceSize()) {
#if 0
      printf("TestKokkos (%d,%d)\n", _N, _Blk);
      printf("dim(A) = { %d, %d, %d }, stride(A) = { %d, %d, %d } \n",
             int(_A.extent(0)), int(_A.extent(1)), int(_A.extent(2)), 
             int(_A.stride(0)), int(_A.stride(1)), int(_A.stride(2)));
      printf("dim(E) = { %d, %d, %d }, stride(E) = { %d, %d, %d } \n",
             int(_E.extent(0)), int(_E.extent(1)), int(_E.extent(2)), 
             int(_E.stride(0)), int(_E.stride(1)), int(_E.stride(2)));
      printf("dim(V) = { %d, %d, %d, %d }, stride(V) = { %d, %d, %d, %d } \n",
             int(_V.extent(0)), int(_V.extent(1)), int(_V.extent(2)), int(_V.extent(3)), 
             int(_V.stride(0)), int(_V.stride(1)), int(_V.stride(2)), int(_V.stride(3)));
      printf("dim(W) = { %d, %d }, stride(W) = { %d, %d } \n",
             int(_W.extent(0)), int(_W.extent(1)), 
             int(_W.stride(0)), int(_W.stride(1)));      
#endif
    }
  };
}

#endif
