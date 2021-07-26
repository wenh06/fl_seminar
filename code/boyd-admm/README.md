# MATLAB scripts for alternating direction method of multipliers from [Boyd's website](https://web.stanford.edu/~boyd/papers/admm/)

## Requirements

1. install [`Matlab kernel for Jupyter`](https://github.com/Calysto/matlab_kernel)

2. install `CVX` from [its official website](http://cvxr.com/cvx/download/)

3. install `L-BFGS-B` from [matlab file exchange](https://www.mathworks.com/matlabcentral/fileexchange/15061-matlab-interface-for-l-bfgs-b)

most probably one has to add `#include <cstring>` in the file `matlabstring.cpp`, and modify the `Makefile`. For example in `Ubuntu20.04`,
```make
# Linux settings.
MEX         = mex
MEXSUFFIX   = mexglx
MATLAB_HOME = /usr/local/MATLAB/R2021a
CXX         = g++
F77         = gfortran
CFLAGS      = -O3 -fPIC -pthread
FFLAGS      = -O3 -fPIC -fexceptions

TARGET = lbfgsb.$(MEXSUFFIX)
OBJS   = solver.o matlabexception.o matlabscalar.o matlabstring.o   \
         matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o \
         lbfgsb.o

CFLAGS += -Wall -ansi -DMATLAB_MEXFILE

all: $(TARGET)

%.o: %.cpp
        $(CXX) $(CFLAGS) -I$(MATLAB_HOME)/extern/include -o $@ -c $^

%.o: %.f
        $(F77) $(FFLAGS) -o $@ -c $^

$(TARGET): $(OBJS)
        $(MEX) -cxx CXX=$(CXX) CC=$(CXX) FC=$(FCC) LD=$(CXX) -lgfortran -lm \
        -O -output $@ $^

clean:
        rm -f *.o $(TARGET)
```
