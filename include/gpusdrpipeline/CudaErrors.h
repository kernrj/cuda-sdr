/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef GPUSDRPIPELINE_CUDAERRORS_H
#define GPUSDRPIPELINE_CUDAERRORS_H

#include <cuda_runtime.h>
#include <gpusdrpipeline/GSErrors.h>
#include <gpusdrpipeline/GSLog.h>

#ifdef DEBUG
#define GPU_SYNC_DEBUG cudaDeviceSynchronize()
#else
#define GPU_SYNC_DEBUG cudaSuccess
#endif

#ifdef DEBUG
#define CHECK_CUDA_OR_RET(msgOnFail__, checkCudaRetOnError__) \
  do {                                                        \
    cudaError_t checkCudaStatus__ = cudaDeviceSynchronize();  \
    if (checkCudaStatus__ != cudaSuccess) {                   \
      gslog(                                                  \
          GSLOG_ERROR,                                        \
          "%s: %s (%d). At %s:%d",                            \
          msgOnFail__,                                        \
          cudaGetErrorName(checkCudaStatus__),                \
          checkCudaStatus__,                                  \
          __FILE__,                                           \
          __LINE__);                                          \
                                                              \
      return checkCudaRetOnError__;                           \
    }                                                         \
  } while (false)

#define CHECK_CUDA_OR_RET_STATUS(msgOnFail__)                \
  do {                                                       \
    cudaError_t checkCudaStatus__ = cudaDeviceSynchronize(); \
    if (checkCudaStatus__ != cudaSuccess) {                  \
      gslog(                                                 \
          GSLOG_ERROR,                                       \
          "%s: %s (%d). At %s:%d",                           \
          msgOnFail__,                                       \
          cudaGetErrorName(checkCudaStatus__),               \
          checkCudaStatus__,                                 \
          __FILE__,                                          \
          __LINE__);                                         \
                                                             \
      return Status_RuntimeError;                            \
    }                                                        \
  } while (false)

#define CHECK_CUDA_OR_RET_RESULT(msgOnFail__)                \
  do {                                                       \
    cudaError_t checkCudaStatus__ = cudaDeviceSynchronize(); \
    if (checkCudaStatus__ != cudaSuccess) {                  \
      gslog(                                                 \
          GSLOG_ERROR,                                       \
          "%s: %s (%d). At %s:%d",                           \
          msgOnFail__,                                       \
          cudaGetErrorName(checkCudaStatus__),               \
          checkCudaStatus__,                                 \
          __FILE__,                                          \
          __LINE__);                                         \
                                                             \
      return {.status = Status_RuntimeError, .value = {}};   \
    }                                                        \
  } while (false)

#define CHECK_CUDA_OR_THROW(msgOnFail__)                     \
  do {                                                       \
    cudaError_t checkCudaStatus__ = cudaDeviceSynchronize(); \
    if (checkCudaStatus__ != cudaSuccess) {                  \
      gslog(                                                 \
          GSLOG_ERROR,                                       \
          "%s: %s (%d). At %s:%d",                           \
          msgOnFail__,                                       \
          cudaGetErrorName(checkCudaStatus__),               \
          checkCudaStatus__,                                 \
          __FILE__,                                          \
          __LINE__);                                         \
                                                             \
      throw std::runtime_error("CUDA Error");                \
    }                                                        \
  } while (false)
#else
#define CHECK_CUDA_OR_RET(msgOnFail__, checkCudaRetOnError__) (void)0
#define CHECK_CUDA_OR_RET_STATUS(msgOnFail__) (void)0
#define CHECK_CUDA_OR_RET_RESULT(msgOnFail__) (void)0
#define CHECK_CUDA_OR_THROW(msgOnFail__) (void)0
#endif

#define SAFE_CUDA_OR_RET(cudaCmd__, safeCudaRetOnFail__)           \
  do {                                                             \
    CHECK_CUDA_OR_RET("Before: " #cudaCmd__, safeCudaRetOnFail__); \
    cudaError_t safeCudaStatus__ = (cudaCmd__);                    \
    if (safeCudaStatus__ != cudaSuccess) {                         \
      gslog(                                                       \
          GSLOG_ERROR,                                             \
          "CUDA error %s: %s. At %s:%d",                           \
          cudaGetErrorName(safeCudaStatus__),                      \
          cudaGetErrorString(safeCudaStatus__),                    \
          __FILE__,                                                \
          __LINE__);                                               \
                                                                   \
      return safeCudaRetOnFail__;                                  \
    }                                                              \
    CHECK_CUDA_OR_RET("After: " #cudaCmd__, safeCudaRetOnFail__);  \
  } while (false)

#define SAFE_CUDA_OR_RET_STATUS(cudaCmd__)           \
  do {                                               \
    CHECK_CUDA_OR_RET_STATUS("Before: " #cudaCmd__); \
    cudaError_t safeCudaStatus__ = (cudaCmd__);      \
    if (safeCudaStatus__ != cudaSuccess) {           \
      gslog(                                         \
          GSLOG_ERROR,                               \
          "CUDA error %s: %s. At %s:%d",             \
          cudaGetErrorName(safeCudaStatus__),        \
          cudaGetErrorString(safeCudaStatus__),      \
          __FILE__,                                  \
          __LINE__);                                 \
                                                     \
      return Status_RuntimeError;                    \
    }                                                \
    CHECK_CUDA_OR_RET_STATUS("After: " #cudaCmd__);  \
  } while (false)

#define SAFE_CUDA_OR_RET_RESULT(cudaCmd__)                 \
  do {                                                     \
    CHECK_CUDA_OR_RET_RESULT("Before: " #cudaCmd__);       \
    cudaError_t safeCudaStatus__ = (cudaCmd__);            \
    if (safeCudaStatus__ != cudaSuccess) {                 \
      gslog(                                               \
          GSLOG_ERROR,                                     \
          "CUDA error %s: %s. At %s:%d",                   \
          cudaGetErrorName(safeCudaStatus__),              \
          cudaGetErrorString(safeCudaStatus__),            \
          __FILE__,                                        \
          __LINE__);                                       \
                                                           \
      return {.status = Status_RuntimeError, .value = {}}; \
    }                                                      \
    CHECK_CUDA_OR_RET_RESULT("After: " #cudaCmd__);        \
  } while (false)

#define SAFE_CUDA_OR_THROW(cudaCmd__)           \
  do {                                          \
    CHECK_CUDA_OR_THROW("Before: " #cudaCmd__); \
    cudaError_t safeCudaStatus__ = (cudaCmd__); \
    if (safeCudaStatus__ != cudaSuccess) {      \
      gslog(                                    \
          GSLOG_ERROR,                          \
          "CUDA error %s: %s. At %s:%d",        \
          cudaGetErrorName(safeCudaStatus__),   \
          cudaGetErrorString(safeCudaStatus__), \
          __FILE__,                             \
          __LINE__);                            \
                                                \
      throw std::runtime_error("CUDA Error");   \
    }                                           \
    CHECK_CUDA_OR_THROW("After: " #cudaCmd__);  \
  } while (false)

#define SAFE_CUDA_WARN_ONLY(cudaCmd__)          \
  do {                                          \
    cudaError_t safeCudaStatus__ = (cudaCmd__); \
    if (safeCudaStatus__ != cudaSuccess) {      \
      gslog(                                    \
          GSLOG_ERROR,                          \
          "CUDA error %s: %s. At %s:%d",        \
          cudaGetErrorName(safeCudaStatus__),   \
          cudaGetErrorString(safeCudaStatus__), \
          __FILE__,                             \
          __LINE__);                            \
    }                                           \
  } while (false)

#endif  // GPUSDRPIPELINE_CUDAERRORS_H
