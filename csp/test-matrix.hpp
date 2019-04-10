#ifndef __TEST_MATRIX_HPP__
#define __TEST_MATRIX_HPP__

#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

namespace TestCSP {
  struct TestMatrix {
    using A_value_type_3d_view_kokkos = Kokkos::View<value_type***, SpT>;  
    using A_value_type_3d_view_mkl    = Kokkos::View<value_type***, Kokkos::LayoutRight,HpT>;  
    
    int _N, _Blk;

    A_value_type_3d_view_kokkos _A_kokkos;
    A_value_type_3d_view_mkl _A_mkl;

    void setRandomMatrix(const int N, const int Blk) {
      _N = N;
      _Blk = Blk; 

      _A_kokkos = A_value_type_3d_view_kokkos("A_mat_kokkos", N, Blk, Blk);
      _A_mkl    = A_value_type_3d_view_mkl   ("A_mat_mkl",    N, Blk, Blk);

      const value_type one(1.0);
      Kokkos::Random_XorShift64_Pool<HpT> random(13245);      
      Kokkos::fill_random(_A_mkl, random, one);

      auto A_kokkos_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _A_kokkos);
      Kokkos::RangePolicy<HpT> policy(0, _N);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
          for (int j=0,jend=A_kokkos_host.extent(1);j<jend;++j) 
            for (int k=0,kend=A_kokkos_host.extent(2);k<kend;++k)
              A_kokkos_host(i,j,k) = _A_mkl(i,j,k);
        });
      Kokkos::deep_copy(_A_kokkos, A_kokkos_host);
    }

    int getBatchsize() const { return _N; }
    int getBlocksize() const { return _Blk; }

    A_value_type_3d_view_kokkos getProblemKokkos() const { return _A_kokkos; }
    A_value_type_3d_view_mkl    getProblemMKL() const { return _A_mkl; }
  };
}

#endif
