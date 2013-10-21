CC = g++
CCFLAGS = -fPIC -O3 -Wall -ffast-math -msse -msse2 -fopenmp
LINKFLAGS = -shared -Wl -fopenmp -lgomp
INPUT = im2col.cpp fastpool.cpp local_response_normalization.cpp neuron.cpp
TARGET = libcpputil.so

# If we are going to use MKL, we include additional flags
MKL_FLAGS = -D DECAF_USE_MKL
MKL_LINK_FLAGS = -lmkl_rt

all: $(INPUT)
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
all_mkl: $(INPUT)
	$(CC) -c $(CCFLAGS) $(MKL_FLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) $(MKL_LINK_FLAGS) -o $(TARGET) *.o
speedtest_lrn: speedtest_lrn.cpp local_response_normalization.cpp
	$(CC) $(CCFLAGS) -lgomp -o speedtest_lrn speedtest_lrn.cpp local_response_normalization.cpp
clean:
	rm *.so
	rm *.o
