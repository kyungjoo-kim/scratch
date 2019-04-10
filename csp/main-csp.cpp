//#define TEST_MKL

/// Kokkos headers
#include "Kokkos_Core.hpp"

typedef Kokkos::DefaultExecutionSpace SpT;
typedef Kokkos::DefaultHostExecutionSpace HpT;
typedef double value_type;

#include "test-matrix.hpp"
#if defined (TEST_MKL)
#include "test-mkl.hpp"
#endif

#include "test-kokkos.hpp"

int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout);                                                       

    /// input arguments parsing
    int N = 1e3; /// # of problems (batch size)
    int Blk = 10; /// dimension of the problem
    for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
    }
    const int niter_beg = -2, niter_end = 3;
    
    printf(" :::: Testing Eigenvalue solver (N = %d, Blk = %d)\n", N, Blk);

    TestCSP::TestMatrix problem;
    problem.setRandomMatrix(N, Blk);
    
#if defined (TEST_MKL)
    {
      TestCSP::TestMKL eig_mkl(N, Blk);
      double t_mkl(0);
      for (int iter=niter_beg;iter<niter_end;++iter) {    
        eig_mkl.setProblem(problem.getProblemMKL());
        const double t = eig_mkl.runTest();
        t_mkl += (iter >= 0)*t;
      }
      printf("MKL           Eigensolver Per Problem Time: %e seconds\n", (t_mkl/double(niter_end*N)));
    }
#endif

    {
      TestCSP::TestKokkos eig_kk(N, Blk);
      double t_kk(0); 
      for (int iter=niter_beg;iter<niter_end;++iter) {          
        eig_kk.setProblem(problem.getProblemKokkos());
        const double t = eig_kk.runTest();
        t_kk += (iter >= 0)*t;
      }
      printf("KokkosBatched Eigensolver Per Problem Time: %e seconds\n", (t_kk/double(niter_end*N)));
    }

  }
  Kokkos::finalize();
  
  return 0;
}
