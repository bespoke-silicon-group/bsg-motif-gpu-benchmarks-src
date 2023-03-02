#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/host_tensor.h>

int main(int argc, char *argv[]) {

  int const m = 512;
  int const n = 512;
  int const k = 512;
  int const batch_count = 64;

  int const lda = m;
  int const ldb = k*batch_count;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;

  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  
  float alpha = 1.25f;
  float beta = -1.25f;


  // host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  //std::vector<float> result_C(count_C);

  // device memory
  float *A;
  float *B;
  float *C;
  cudaMalloc(&A, count_A * sizeof(float));
  cudaMalloc(&B, count_B * sizeof(float));
  cudaMalloc(&C, count_C * sizeof(float));
    
  // copy from host to memory
  cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({ 
    {m,n,k},
    {A, lda},
    batch_stride_A,
    {B, ldb},
    batch_stride_B,
    {C, ldc},
    batch_stride_C,
    {C, ldc},
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

    if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}
