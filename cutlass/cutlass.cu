#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main(int argc, char *argv[]) {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    float,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    float,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassSimt,            // tag indicating Tensor Cores
    cutlass::arch::Sm75                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //
  int M = 1024;
  int N = 1024;
  int K = 1024;
  
  if(argc >= 2)
  	M = N = K = atoi(argv[1]);
  	
  printf("Square matrix size: %d\n", M);

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> C({M, N});

  float const *ptrA = A.device_data();
  float const *ptrB = B.device_data();
  float const *ptrC = C.device_data();
  float       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  return 0;
}
