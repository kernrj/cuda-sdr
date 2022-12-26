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

#include <cuComplex.h>

#include "QuadDemod.h"
#include "cuComplexOperatorOverloads.cuh"

__global__ static void k_quadDemod(const cuComplex* input, float* output, float gain) {
  uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

  const cuComplex m = input[index + 1] * cuConjf(input[index]);

  output[index] = gain * atan2f(m.y, m.x);
}
