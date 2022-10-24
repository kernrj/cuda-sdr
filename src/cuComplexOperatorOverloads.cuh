/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_CUCOMPLEXOPERATOROVERLOADS_H_
#define SDRTEST_SRC_CUCOMPLEXOPERATOROVERLOADS_H_

#include <cuda_runtime_api.h>

__forceinline__ __host__ __device__ cuComplex
operator*(const cuComplex c1, const cuComplex c2) {
  return cuCmulf(c1, c2);
}

__forceinline__ __host__ __device__ cuComplex
operator*(const cuComplex c, const float r) {
  return make_cuComplex(cuCrealf(c) * r, cuCimagf(c) * r);
}

__forceinline__ __host__ __device__ cuComplex
operator*(const float r, const cuComplex c) {
  return c * r;
}

__forceinline__ __host__ __device__ cuComplex
operator/(const cuComplex c1, const cuComplex c2) {
  return cuCdivf(c1, c2);
}

__forceinline__ __host__ __device__ cuComplex
operator/(const cuComplex c, const float r) {
  return make_cuComplex(cuCrealf(c) / r, cuCimagf(c) / r);
}

__forceinline__ __host__ __device__ cuComplex
operator/(const float r, const cuComplex c) {
  return make_cuComplex(r / cuCrealf(c), r / cuCimagf(c));
}

__forceinline__ __host__ __device__ cuComplex
operator+(const cuComplex c1, const cuComplex c2) {
  return cuCaddf(c1, c2);
}

__forceinline__ __host__ __device__ cuComplex
operator+(const cuComplex c, const float r) {
  return make_cuComplex(cuCrealf(c) + r, cuCimagf(c));
}

__forceinline__ __host__ __device__ cuComplex
operator+(const float r, const cuComplex c) {
  return c + r;
}

__forceinline__ __host__ __device__ cuComplex& operator+=(
    cuComplex& lhs,
    const cuComplex rhs) {
  lhs.x += rhs.x;
  lhs.y += rhs.y;

  return lhs;
}

template <class T>
__forceinline__ __host__ __device__ T zero() {
  return 0;
}

template <>
__forceinline__ __host__ __device__ cuComplex zero<cuComplex>() {
  return make_cuComplex(0, 0);
}

#endif  // SDRTEST_SRC_CUCOMPLEXOPERATOROVERLOADS_H_
