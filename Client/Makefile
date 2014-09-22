#!/bin/sh

CC = g++
CU = nvcc
CUFLAG = -lcudart
#FLAG = -g -G
FLAG = 
INC = -I/usr/local/cuda-6.0/include/
CUDAFLAG = -gencode arch=compute_50,code=compute_50\
           -gencode arch=compute_35,code=compute_35\
           -gencode arch=compute_20,code=compute_20

OBJECT=caffe
SOURCE=main.o parsing.o initMatrix.o print.o load.o check.o boundary.o \
       cu_svd_val.o cu_util.o cu_small_val.o cu_huge_val.o cu_svd_vec.o

$(OBJECT):$(SOURCE)
	$(CU) $(FLAG) $(INC) $+ $(CUFLAG) $(CUDAFLAG) -o $@

%.o:%.cu
	$(CU) $(FLAG) $(INC) $(CUDAFLAG) -o $@ -c $<
%.o:%.cpp
	$(CC) $(INC) -c $< -o $@

clean:
	rm -rf *.o $(OBJECT)
