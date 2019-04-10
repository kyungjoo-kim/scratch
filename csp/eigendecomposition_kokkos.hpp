#ifndef EIGENDECOMPOSITION_KOKKOS_CSP
#define EIGENDECOMPOSITION_KOKKOS_CSP

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Eigendecomposition_Decl.hpp"
#include "KokkosBatched_Eigendecomposition_Serial_Impl.hpp"

//#define EIGENDECOMPOSITION_KOKKOS_CSP_TIMER

template<typename SpT>
struct EigendecompositionKokkos {
public:

  ///
  /// Interface via Kokkos views
  ///
  /// A(N, Blk, Blk)
  /// er(N, Blk)
  /// ei(N, Blk)
  /// UL(N, Blk, Blk)
  /// UR(N, Blk, Blk)
  /// W(N, wsize) where wsize minimum is 2*Blk*Blk + Blk*5.
  template<typename AViewType,
           typename EViewType,
           typename UViewType,
           typename WViewType>
  inline
  static double invoke(/* const */ AViewType &_A,
                       const EViewType &_er, const EViewType &_ei,
                       const UViewType &_UL, const UViewType &_UR,
                       const WViewType &_W) {
#if defined(EIGENDECOMPOSITION_KOKKOS_CSP_TIMER)
    Kokkos::Impl::Timer timer;
    timer.reset();
#endif

    static_assert(AViewType::rank == 3, "A is not rank 3");
    static_assert(EViewType::rank == 2, "Eigenvalue views are not rank 2");
    static_assert(UViewType::rank == 3, "Eigenvector views are not rank 3");
    static_assert(WViewType::rank == 2, "Workspace view is not rank 2");

    /// # of eigen problems
    const int N   = _A.extent(0);
    assert(N == _er.extent(0) && "er extent(0) does not match to N");
    assert(N == _ei.extent(0) && "ei extent(0) does not match to N");
    assert(N == _UL.extent(0) && "UL extent(0) does not match to N");
    assert(N == _UR.extent(0) && "UR extent(0) does not match to N");
    assert(N == _W.extent(0)  && "W extent(0) does not match to N");

    /// eigen problem size
    const int Blk = _A.extent(1);
    assert(Blk == _A.extent(2)  && "A's problems are not square");
    assert(Blk == _er.extent(1) && "er does not match to Blk");
    assert(Blk == _ei.extent(1) && "ei does not match to Blk");
    assert(Blk == _UL.extent(1) && "UL (extent 1) does not match to Blk");
    assert(Blk == _UL.extent(2) && "UL (extent 2) does not match to Blk");
    assert(Blk == _UR.extent(1) && "UL (extent 1) does not match to Blk");
    assert(Blk == _UR.extent(2) && "UL (extent 2) does not match to Blk");

    /// workspace for each problem
    const int wsize = _W.extent(1);
    assert(wsize >= (2*Blk*Blk + Blk*5) && "Workspace is too small");

    /// parallel policy
    const Kokkos::RangePolicy<SpT> policy(0, N);

    /// subview patterns
    auto A  = Kokkos::subview(_A,  0, Kokkos::ALL(), Kokkos::ALL());
    auto er = Kokkos::subview(_er, 0, Kokkos::ALL());
    auto ei = Kokkos::subview(_ei, 0, Kokkos::ALL());
    auto UL = Kokkos::subview(_UL, 0, Kokkos::ALL(), Kokkos::ALL());
    auto UR = Kokkos::subview(_UR, 0, Kokkos::ALL(), Kokkos::ALL());
    auto W  = Kokkos::subview(_W,  0, Kokkos::ALL());

    const int as0 = A.stride(0), as1 = A.stride(1);
    const int es = er.stride(0);
    const int us0 = UL.stride(0), us1 = UL.stride(1);    

    /// parallel pattern
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int &prob) {
        KokkosBatched::Experimental
          ::SerialEigendecompositionInternal
          ::invoke(Blk, 
                   &_A (prob, 0, 0), as0, as1,
                   &_er(prob, 0   ), es,
                   &_ei(prob, 0   ), es,
                   &_UL(prob, 0, 0), us0, us1,
                   &_UR(prob, 0, 0), us0, us1,
                   &_W (prob, 0   ), wsize);
      });

#if defined(EIGENDECOMPOSITION_KOKKOS_CSP_TIMER)
    const double t = timer.seconds();
    printf("EigendecompositionKokkos::view interface time = %e\n", t);
#else
    const double t = 0;
#endif
    return t;
  }

  ///
  /// Interface via std::vector multi dimensional arrays
  ///
  static double invoke(const int nprob,
                       const int nrow,
                       const int ncol,
                       const std::vector<std::vector<std::vector<double> > > &matrix_in,
                       /* */             std::vector<std::vector<double> >   &eig_val_real,
                       /* */             std::vector<std::vector<double> >   &eig_val_imag,
                       /* */ std::vector<std::vector<std::vector<double> > > &eig_vec_L,
                       /* */ std::vector<std::vector<std::vector<double> > > &eig_vec_R) {
#if defined(EIGENDECOMPOSITION_KOKKOS_CSP_TIMER)
    Kokkos::Impl::Timer timer;
    timer.reset();
#endif

    assert(nrow == ncol && "nrow is not same as ncol; eigendecomposition requires a square matrix");
    Kokkos::View<double***,SpT> A ("A",  nprob, nrow, ncol);
    Kokkos::View<double**, SpT> er("er", nprob, nrow);
    Kokkos::View<double**, SpT> ei("ei", nprob, nrow);
    Kokkos::View<double***,SpT> UL("UL", nprob, nrow, ncol);
    Kokkos::View<double***,SpT> UR("UR", nprob, nrow, ncol);
    const int wsize = 2*nrow*nrow + nrow*5;
    Kokkos::View<double**, Kokkos::LayoutRight, SpT> W ("W",  nprob, wsize);

    /// various parallel policy
    using md_rank3_range_policy_type = Kokkos::Experimental::MDRangePolicy
      <Kokkos::DefaultHostExecutionSpace,Kokkos::Experimental::Rank<3>, Kokkos::IndexType<int> >;
    const md_rank3_range_policy_type md_policy_rank3( {0, 0, 0}, {nprob, nrow, ncol} );

    using md_rank2_range_policy_type = Kokkos::Experimental::MDRangePolicy
      <Kokkos::DefaultHostExecutionSpace,Kokkos::Experimental::Rank<2>, Kokkos::IndexType<int> >;
    const md_rank2_range_policy_type md_policy_rank2( {0, 0}, {nprob, nrow} );

    /// pack the data into a host view
    auto Ah = Kokkos::create_mirror_view(A);
    Kokkos::parallel_for(md_policy_rank3, [&](const int i, const int j, const int k) {
        Ah(i,j,k) = matrix_in.at(i).at(j).at(k);
      });

    /// copy from host to device
    Kokkos::deep_copy(A, Ah);

    /// parallel execution with a range policy
    const double t_view_interf = invoke(A, er, ei, UL, UR, W);

    /// copy back the result to host std vectors
    auto erh = Kokkos::create_mirror_view(er); Kokkos::deep_copy(erh, er);
    auto eih = Kokkos::create_mirror_view(ei); Kokkos::deep_copy(eih, ei);
    auto ULh = Kokkos::create_mirror_view(UL); Kokkos::deep_copy(ULh, UL);
    auto URh = Kokkos::create_mirror_view(UR); Kokkos::deep_copy(URh, UR);

    Kokkos::parallel_for(md_policy_rank2, [&](const int i, const int j) {
        eig_val_real.at(i).at(j) = erh(i,j);
        eig_val_imag.at(i).at(j) = eih(i,j);
      });
    Kokkos::parallel_for(md_policy_rank3, [&](const int i, const int j, const int k) {
        eig_vec_L.at(i).at(j).at(k) = ULh(i,j,k);
        eig_vec_R.at(i).at(j).at(k) = URh(i,j,k);
      });

#if defined(EIGENDECOMPOSITION_KOKKOS_CSP_TIMER)
    const double t = timer.seconds();
    printf("EigendecompositionKokkos::total std vector interface time = %e, computation = %e, overhead = %e\n",
           t, t_view_interf, (t-t_view_interf));
#else
    const double t = 0;
#endif
    return t;
  }

};

#endif  //end of header guard
