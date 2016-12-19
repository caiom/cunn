#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#ifdef CUDA_HALF_TENSOR
void THNN_CudaHalfLinearSparse_updateOutput(
           THCState * state,
           THCudaHalfTensor * inputT,
           THCudaHalfTensor * outputT,
           THCudaHalfTensor * weightT,
           THCudaIntTensor * rowsT,
           THCudaIntTensor * colsT,
           THCudaIntTensor * iRowStart,
           int nRowWeight,
           int nColWeight){
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfLinearSparse_updateGradInput(
           THCState * state,
           THCudaHalfTensor * inputDummy,
           THCudaHalfTensor * inputT,
           THCudaHalfTensor * outputT,
           THCudaHalfTensor * weightT,
           THCudaIntTensor * rowsT,
           THCudaIntTensor * colsT,
           int nRowWeight,
           int nColWeight,
           THCudaIntTensor * iRowStartT){
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfLinearSparse_accGradParameters(
           THCState * state,
           THCudaHalfTensor * inputT,
           THCudaHalfTensor * gradOutT,
           THCudaHalfTensor * weightT,
           THCudaIntTensor * rowsT,
           THCudaIntTensor * colsT,
           int nnz,
           float scale){
  THError("THCudaHalfTensor not supported with SparseLinear");
}
#endif

#include "generic/LinearSparse.cu"
#include "THCGenerateFloatType.h"
#include "generic/LinearSparse.cu"
#include "THCGenerateDoubleType.h"


