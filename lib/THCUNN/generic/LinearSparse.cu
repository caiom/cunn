#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LinearSparse.cu"
#else

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS(n) (int)THCCeilDiv(n, (long) BLOCK_SIZE)

void THNN_(LinearSparse_updateOutput)(
           THCState * state,
           THCTensor * inputT,
           THCTensor * outputT,
           THCTensor * weightT,
           THCudaIntTensor * rowsT,
           THCudaIntTensor * colsT,
           THCudaIntTensor * iRowStart,
           int nRowWeight,
           int nColWeight) {

  THAssert(THCudaTensor_checkGPU(state, 6, inputT, outputT, weightT,
                                 rowsT, colsT, iRowStart));
  //THArgCheck(inputT->nDimension == 2, 4, "Input should be 2D");
  //THArgCheck(outputT->nDimension == 2, 4, "output tensor must be 2D");

  int nRowInput = inputT->size[0];
  int nColInput = inputT->size[1];

  int nnz = weightT->size[0];

  char transa = 'n';
  char transb = 'n';

  real alpha = 1.0;
  real beta = 0.0;

  if(inputT->nDimension == 1)
  {
    beta = 1.0f;

    #ifdef THC_REAL_IS_FLOAT
    THCusparse_Scsrmv(state,
    #elif defined(THC_REAL_IS_DOUBLE)
    THCusparse_Dcsrmv(state,
    #endif
                     transa, nRowWeight, nColWeight, nnz, alpha, THCTensor_(data)(state, weightT), THCudaIntTensor_data(state, iRowStart), 
                     THCudaIntTensor_data(state, colsT), THCTensor_(data)(state, inputT), beta, THCTensor_(data)(state, outputT));
  }
  else
  {

    #ifdef THC_REAL_IS_FLOAT
    THCusparse_Scsrmm2(state,
    #elif defined(THC_REAL_IS_DOUBLE)
    THCusparse_Dcsrmm2(state,
    #endif
                      transa, transb, nRowWeight, nRowInput, nColWeight, nnz, alpha, 
                      THCTensor_(data)(state, weightT), THCudaIntTensor_data(state, iRowStart), 
                      THCudaIntTensor_data(state, colsT), THCTensor_(data)(state, inputT), nColInput, beta, 
                      THCTensor_(data)(state, outputT), nRowWeight);
  }
}

void THNN_(LinearSparse_updateGradInput)(
           THCState * state,
           THCTensor * inputDummy,
           THCTensor * inputT,
           THCTensor * outputT,
           THCTensor * weightT,
           THCudaIntTensor * rowsT,
           THCudaIntTensor * colsT,
           int nRowWeight,
           int nColWeight,
           THCudaIntTensor * iRowStartT) {

  THAssert(THCudaTensor_checkGPU(state, 6, inputT, outputT, weightT,
                                 rowsT, colsT, iRowStartT));

  //int nRowInput = inputT->size[0];
  int nColInput = inputT->size[1];
  int nRowOutput = outputT->size[0];
  int nColOutput = outputT->size[1];

  int nnz = weightT->size[0];

  char transa = 't';
  char transb = 'n';

  real alpha = 1.0;
  real beta = 0.0;

  if(inputT->nDimension == 1)
  {

    #ifdef THC_REAL_IS_FLOAT
    THCusparse_Scsrmv(state,
    #elif defined(THC_REAL_IS_DOUBLE)
    THCusparse_Dcsrmv(state,
    #endif
                     transa, nRowWeight, nColWeight, nnz, alpha, THCTensor_(data)(state, weightT), THCudaIntTensor_data(state, iRowStartT), 
                     THCudaIntTensor_data(state, colsT), THCTensor_(data)(state, inputT), beta, THCTensor_(data)(state, outputT));
  }
  else
  {
 
    #ifdef THC_REAL_IS_FLOAT
    THCusparse_Scsrmm2(state,
    #elif defined(THC_REAL_IS_DOUBLE)
    THCusparse_Dcsrmm2(state,
    #endif
                    transa, transb, nRowWeight, nRowOutput, nColWeight, nnz, alpha, 
                    THCTensor_(data)(state, weightT), THCudaIntTensor_data(state, iRowStartT), 
                    THCudaIntTensor_data(state, colsT), THCTensor_(data)(state, inputT), nColInput, beta, 
                    THCTensor_(data)(state, outputT), nColOutput); 
   }
}

__global__ void linearSparse2DimAccGradParameters(int nnz, real * input, real * gradOut, real * weight, int * rows, int * cols, real scale, int batchSize, int strideGrad, int strideInp)
{
  int j;
  CUDA_KERNEL_LOOP(i, nnz)
  {
     real sum = 0.0;

     //int row = 0;

     //while(rowStart[row+1] <= i)
     //  row++;

     for(j=0; j<batchSize; j++)
     {
	 sum += gradOut[j*strideGrad + rows[i]] * input[j*strideInp + cols[i]];
     }

     weight[i] += sum*scale;
  }
}

__global__ void linearSparse1DimAccGradParameters(int nnz, real * input, real * gradOut, real * weight, int * rows, int * cols, real scale)
{
  CUDA_KERNEL_LOOP(i, nnz)
  {
    // int row = 0;
    // while(i < rowStart[row+1])
    //   row++;
     
     weight[i] += gradOut[rows[i]] * input[cols[i]] * scale;
  }
}

void THNN_(LinearSparse_accGradParameters)(
           THCState * state,
           THCTensor * inputT,
           THCTensor * gradOutT,
           THCTensor * weightT,
           THCudaIntTensor * rowsT,
           THCudaIntTensor * colsT,
           int nnz,
           float scale) {

  THAssert(THCudaTensor_checkGPU(state, 5, inputT, gradOutT, weightT,
                                 rowsT, colsT));

  int nRowInput = inputT->size[0];
  int nColInput = inputT->size[1];
  //int nRowOutput = gradOutT->size[0];
  int nColOutput = gradOutT->size[1];
  long n = nnz;

  real scale_r = (real) scale;

  if(inputT->nDimension == 1)
  {

    linearSparse1DimAccGradParameters<<<NUM_BLOCKS(n), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        n, THCTensor_(data)(state, inputT), THCTensor_(data)(state, gradOutT), THCTensor_(data)(state, weightT),
        THCudaIntTensor_data(state, rowsT), THCudaIntTensor_data(state, colsT),  scale_r);

  }
  else
  {
      linearSparse2DimAccGradParameters<<<NUM_BLOCKS(n), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        n, THCTensor_(data)(state, inputT), THCTensor_(data)(state, gradOutT), THCTensor_(data)(state, weightT),
        THCudaIntTensor_data(state, rowsT), THCudaIntTensor_data(state, colsT),  scale_r, nRowInput, nColOutput, nColInput);
   }

  
}

#endif


