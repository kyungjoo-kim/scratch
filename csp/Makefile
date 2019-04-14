#blake
KOKKOS_INSTALL_PATH=$(HOME)/Work/lib/kokkos/install/blake/release
KOKKOSKERNELS_INSTALL_PATH=$(HOME)/Work/lib/kokkoskernels/install/blake/release
EXTRA_CXXFLAGS = -DTEST_MKL -DTEST_CHECK

#bowman
#KOKKOS_INSTALL_PATH=$(HOME)/Work/lib/kokkos/install/bowman/release
#KOKKOSKERNELS_INSTALL_PATH=$(HOME)/Work/lib/kokkoskernels/install/bowman/release
#EXTRA_CXXFLAGS = -DTEST_MKL -DTEST_CHECK

#kokkos dev 2
#KOKKOS_INSTALL_PATH=$(HOME)/Work/lib/kokkos/install/kokkos-dev-2/release
#KOKKOSKERNELS_INSTALL_PATH=$(HOME)/Work/lib/kokkoskernels/install/kokkos-dev-2/release
#EXTRA_CXXFLAGS = -DTEST_CHECK

#white	
#KOKKOS_INSTALL_PATH=$(HOME)/Work/lib/kokkos/install/white/release-p100
#KOKKOSKERNELS_INSTALL_PATH=$(HOME)/Work/lib/kokkoskernels/install/white/release-p100
#EXTRA_CXXFLAGS = -DTEST_CHECK -DTEST_MAGMA

SRC = $(wildcard *.cpp)
HDR = $(wildcard *.hpp)

include $(KOKKOS_INSTALL_PATH)/Makefile.kokkos
include $(KOKKOSKERNELS_INSTALL_PATH)/Makefile.kokkos-kernels

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = $(NVCC_WRAPPER) -g -lineinfo
else
CXX = icpc -g -mkl -O3
endif

CPPFLAGS =-I.
CXXFLAGS = 
LINK = ${CXX} -g

LINKFLAGS = 

DEPFLAGS = -M

LIB = 


this : $(TEST).o $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS)  
	$(LINK) $(KOKKOS_LDFLAGS) $(KOKKOSKERNELS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(TEST).o $(KOKKOSKERNELS_LIB) $(KOKKOS_LIBS) $(LIB) -o $(TEST).exe

clean: 
	rm -f *.o *.cuda *.host *~ *.exe

# Compilation rules

%.o:%.cpp 
	$(CXX) $(KOKKOSKERNELS_CPPFLAGS) $(KOKKOS_CPPFLAGS) $(CPPFLAGS) $(KOKKOSKERNELS_CXXFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -c $< -o $@
