// CUDA runtime API compatibility layer for ROCm on Windows.
//
// PyTorch ships raw (non-hipified) c10/cuda and ATen/cuda headers, which
// #include <cuda_runtime_api.h> and reference the CUDA runtime types and
// functions directly. On Linux ROCm installs, /opt/rocm/include provides a
// complete compat header. The Windows ROCm SDK ships only a stub that defines
// cudaDeviceProp and nothing else, so we provide the full mapping here.
//
// All aliases point at HIP equivalents; every enum layout matches the CUDA
// one (verified against HIP headers), so no value translation is required.
// Inline wrappers forward directly to the HIP runtime.
#pragma once

#include <hip/hip_runtime_api.h>
#include <hip/library_types.h>
#include <hip/driver_types.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#include <hipsolver/hipsolver.h>

// ---- scalar/error types ----
typedef hipError_t cudaError_t;
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipMemoryType cudaMemoryType;

// ---- cuBLAS / cuSPARSE / cuSOLVER status aliases (SDK shims only typedef the
// handles; torch's Exceptions.h/CUDAContextLight.h need the status types too) ----
typedef hipblasStatus_t cublasStatus_t;
typedef hipsparseStatus_t cusparseStatus_t;
typedef hipsolverDnHandle_t cusolverDnHandle_t;
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_LICENSE_ERROR
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_STATUS_NOT_INITIALIZED HIPSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_ALLOC_FAILED HIPSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_INVALID_VALUE HIPSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_ARCH_MISMATCH HIPSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_MAPPING_ERROR HIPSPARSE_STATUS_MAPPING_ERROR
#define CUSPARSE_STATUS_EXECUTION_FAILED HIPSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_INTERNAL_ERROR HIPSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_ZERO_PIVOT HIPSPARSE_STATUS_ZERO_PIVOT

// ---- stream ----
typedef hipStream_t cudaStream_t;

typedef enum hipStreamCaptureMode cudaStreamCaptureMode;
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed

typedef enum hipStreamCaptureStatus cudaStreamCaptureStatus;
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated

// ---- event ----
typedef hipEvent_t cudaEvent_t;
typedef hipIpcEventHandle_t cudaIpcEventHandle_t;

#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventDefault hipEventDefault
#define cudaEventRecordDefault (0x0u)
#define cudaEventRecordExternal (0x20000000u)
#define cudaEventWaitDefault (0x0u)
#define cudaEventWaitExternal (0x20000000u)

// ---- memory ----
typedef hipMemcpyKind cudaMemcpyKind;
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeManaged hipMemoryTypeManaged

typedef hipPointerAttribute_t cudaPointerAttributes;

// ---- graphs ----
typedef hipGraph_t cudaGraph_t;
typedef hipGraphExec_t cudaGraphExec_t;

// ---- data types (value layouts are identical to CUDA) ----
typedef hipDataType cudaDataType;
#define CUDA_R_16F HIP_R_16F
#define CUDA_C_16F HIP_C_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_C_32F HIP_C_32F
#define CUDA_R_64F HIP_R_64F
#define CUDA_C_64F HIP_C_64F
#define CUDA_R_8I HIP_R_8I
#define CUDA_C_8I HIP_C_8I
#define CUDA_R_8U HIP_R_8U
#define CUDA_C_8U HIP_C_8U
#define CUDA_R_32I HIP_R_32I
#define CUDA_C_32I HIP_C_32I
#define CUDA_R_32U HIP_R_32U
#define CUDA_C_32U HIP_C_32U
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_C_16BF HIP_C_16BF
#define CUDA_R_16I HIP_R_16I
#define CUDA_C_16I HIP_C_16I

// ---- runtime API wrappers ----
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady

static inline cudaError_t cudaGetDevice(int* device) { return hipGetDevice(device); }
static inline cudaError_t cudaSetDevice(int device) { return hipSetDevice(device); }
static inline cudaError_t cudaGetDeviceCount(int* count) { return hipGetDeviceCount(count); }
static inline cudaError_t cudaDeviceSynchronize(void) { return hipDeviceSynchronize(); }
static inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  return hipMemGetInfo(free, total);
}
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  return hipGetDeviceProperties(prop, device);
}

typedef enum hipDeviceAttribute_t cudaDeviceAttr;
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

static inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  return hipDeviceGetAttribute(value, attr, device);
}
static inline cudaError_t cudaMalloc(void** ptr, size_t size) { return hipMalloc(ptr, size); }
static inline cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  return hipMallocAsync(ptr, size, stream);
}
static inline cudaError_t cudaFree(void* ptr) { return hipFree(ptr); }
static inline cudaError_t cudaFreeHost(void* ptr) { return hipHostFree(ptr); }
static inline cudaError_t cudaHostFree(void* ptr) { return hipHostFree(ptr); }
static inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                          cudaMemcpyKind kind, cudaStream_t stream) {
  return hipMemcpyAsync(dst, src, count, kind, stream);
}
static inline cudaError_t cudaMemcpyAsyncPeer(void* dst, int dst_device, const void* src,
                                              int src_device, size_t count, cudaStream_t stream) {
  return hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
}

static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return hipStreamSynchronize(stream);
}
static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                              unsigned int flags = 0) {
  return hipStreamWaitEvent(stream, event, flags);
}
static inline cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
  return hipStreamGetPriority(stream, priority);
}
static inline cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
  return hipDeviceGetStreamPriorityRange(least, greatest);
}
static inline cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
  return hipThreadExchangeStreamCaptureMode(mode);
}
static inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                                cudaStreamCaptureStatus* status) {
  return hipStreamIsCapturing(stream, status);
}
static inline cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                                   cudaStreamCaptureStatus* status,
                                                   uint64_t* capture_id = nullptr) {
  return hipStreamGetCaptureInfo(stream, status, capture_id);
}
static inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
  return hipStreamEndCapture(stream, graph);
}

static inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
  return hipEventCreate(event);
}
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  return hipEventCreateWithFlags(event, flags);
}
static inline cudaError_t cudaEventDestroy(cudaEvent_t event) { return hipEventDestroy(event); }
static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  return hipEventElapsedTime(ms, start, end);
}
static inline cudaError_t cudaEventQuery(cudaEvent_t event) { return hipEventQuery(event); }
static inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
  return hipEventRecord(event, stream);
}
static inline cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                   unsigned int flags) {
  return hipEventRecordWithFlags(event, stream, flags);
}
static inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return hipEventSynchronize(event);
}
static inline cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
  return hipIpcGetEventHandle(handle, event);
}
static inline cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
  return hipIpcOpenEventHandle(event, handle);
}

static inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes,
                                                   const void* ptr) {
  return hipPointerGetAttributes(attributes, ptr);
}

static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* graph_exec, cudaGraph_t graph,
                                               unsigned long long flags = 0) {
  return hipGraphInstantiateWithFlags(graph_exec, graph, flags);
}

static inline const char* cudaGetErrorString(cudaError_t error) {
  return hipGetErrorString(error);
}
static inline cudaError_t cudaGetLastError(void) { return hipGetLastError(); }

// ---- function attributes (cudaFuncSetAttribute path) ----
typedef enum hipFuncAttribute cudaFuncAttribute_t;
typedef struct hipFuncAttributes cudaFuncAttributes;
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxThreadsPerBlock hipFuncAttributeMaxThreadsPerBlock
#define cudaFuncAttributeBinarySize hipFuncAttributeBinarySize
#define cudaFuncAttributePreferredSharedMemoryCarrier hipFuncAttributeMaxDynamicSharedMemorySize

static inline cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute_t attr,
                                               size_t value) {
  return hipFuncSetAttribute(func, static_cast<hipFuncAttribute>(attr), static_cast<int>(value));
}

// ---- CUDA Runtime API 1:1 aliases for every CUDA Runtime call that the
// (non-hipified) PyTorch c10/cuda and ATen/cuda headers, plus the sgl_kernel
// device sources, emit. HIP exposes an identical function for each, so a
// direct object-like macro mapping is both safe and complete. Names already
// provided as inline wrappers above are intentionally excluded. ----
#define cudaStreamQuery                  hipStreamQuery
#define cudaStreamCreate                 hipStreamCreate
#define cudaStreamCreateWithFlags        hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority     hipStreamCreateWithPriority
#define cudaStreamDestroy                hipStreamDestroy
#define cudaStreamDestroyAsync           hipStreamDestroyAsync
#define cudaStreamAddCallback            hipStreamAddCallback
#define cudaStreamAttachMem              hipStreamAttachMem
#define cudaFuncGetAttributes            hipFuncGetAttributes
#define cudaMemset                       hipMemset
#define cudaMemsetAsync                  hipMemsetAsync
#define cudaIpcGetMemHandle              hipIpcGetMemHandle
#define cudaIpcCloseMemHandle            hipIpcCloseMemHandle
#define cudaIpcOpenMemHandle             hipIpcOpenMemHandle
#define cudaDriverGetVersion             hipDriverGetVersion
#define cudaRuntimeGetVersion            hipRuntimeGetVersion
#define cudaDeviceGetLimit               hipDeviceGetLimit
#define cudaDeviceSetLimit               hipDeviceSetLimit
#define cudaDeviceReset                  hipDeviceReset
#define cudaDeviceGetPrimaryCtx          hipDeviceGetPrimaryCtx
#define cudaDevicePrimaryCtxRelease      hipDevicePrimaryCtxRelease
#define cudaDevicePrimaryCtxRetain       hipDevicePrimaryCtxRetain
#define cudaDevicePrimaryCtxSetFlags     hipDevicePrimaryCtxSetFlags
#define cudaPeekAtLastError              hipPeekAtLastError
#define cudaDeviceGet                    hipDeviceGet
#define cudaCtxGetFlags                  hipCtxGetFlags
#define cudaMemcpy                       hipMemcpy
#define cudaMemcpyPeer                   hipMemcpyPeer
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaGridDependencySynchronize    hipDeviceSynchronize

// ---- API renames: newer torch moved these from at::cuda into c10::cuda.
// Pull CUDAStream.h first so c10::cuda::getCurrentCUDAStream is declared, then
// re-expose the old at::cuda:: spelling the sgl_kernel sources use. ----
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
namespace at {
namespace cuda {
using ::c10::cuda::CUDAStream;
using ::c10::cuda::CUDAGuard;
using ::c10::cuda::OptionalCUDAGuard;
using ::c10::cuda::current_device;
inline ::c10::cuda::CUDAStream getCurrentCUDAStream(int device_index = -1) {
  return ::c10::cuda::getCurrentCUDAStream(device_index);
}} // namespace cuda
} // namespace at
