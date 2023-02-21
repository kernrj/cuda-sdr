/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
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

#include "filters/FilterFactories.h"

GS_C_LINKAGE Result<Filter> createFilter(const char* name, const char* jsonParameters) noexcept {
  return ERR_RESULT(Status_UnknownError);
}

GS_C_LINKAGE Result<Source> createSource(const char* name, const char* jsonParameters) noexcept {
  return ERR_RESULT(Status_UnknownError);
}

GS_C_LINKAGE Result<Source> createSink(const char* name, const char* jsonParameters) noexcept {
  return ERR_RESULT(Status_UnknownError);
}

GS_C_LINKAGE bool hasFilterFactory(const char* name) noexcept { return false; }

GS_C_LINKAGE bool hasSourceFactory(const char* name) noexcept { return false; }

GS_C_LINKAGE bool hasSinkFactory(const char* name) noexcept { return false; }

GS_C_LINKAGE void registerFilterFactory(
    const char* name,
    void* context,
    Filter* (*filterFactory)(const char* jsonParamters)) noexcept {}

GS_C_LINKAGE void registerSourceFactory(
    const char* name,
    void* context,
    Source* (*filterFactory)(const char* jsonParamters)) noexcept {}

GS_C_LINKAGE void registerSinkFactory(
    const char* name,
    void* context,
    Sink* (*sinkFactory)(const char* jsonParamters)) noexcept {}
