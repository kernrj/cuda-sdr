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

#ifndef GPUSDR_GSERRORS_H
#define GPUSDR_GSERRORS_H

#include <gpusdrpipeline/GSLog.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Example: SSTREAM("hello " << username << " you are " << age << " years old.");
#define SSTREAM(x) dynamic_cast<std::ostringstream&&>(std::ostringstream() << x).str()

#define ERRNO_THROW(cmdReturningErrno__)                                                        \
  do {                                                                                          \
    int errnoValue__ = cmdReturningErrno__;                                                     \
    if (errnoValue__ == ENOMEM) {                                                               \
      throw std::bad_alloc();                                                                   \
    } else if (errnoValue__ != 0) {                                                             \
      throw std::runtime_error(SSTREAM(strerror(errnoValue__) << " (" << errnoValue__ << ')')); \
    }                                                                                           \
  } while (0)

// Example: GS_FAIL("Illegal argument: " << argument);
#define GS_FAIL(x)                                                            \
  do {                                                                        \
    auto __msg = SSTREAM(x);                                                  \
    std::cerr << __msg << " at " << __FILE__ << ':' << __LINE__ << std::endl; \
    abort();                                                                  \
  } while (false)

#ifdef DEBUG
#define GS_DEBUG_FAIL(x)                                                      \
  do {                                                                        \
    auto __msg = SSTREAM(x);                                                  \
    std::cerr << __msg << " at " << __FILE__ << ':' << __LINE__ << std::endl; \
    abort();                                                                  \
  } while (false)
#else
#define GS_DEBUG_FAIL(x)                                                      \
  do {                                                                        \
    auto __msg = SSTREAM(x);                                                  \
    std::cerr << __msg << " at " << __FILE__ << ':' << __LINE__ << std::endl; \
  } while (false)
#endif

#define GS_REQUIRE_OR_RET_STATUS(requireCmd__, messageOnFalse__)                                                   \
  do {                                                                                                             \
    if (!(requireCmd__)) {                                                                                         \
      gslogd("Expression must be true [%s] - %s - at %s:%d", #requireCmd__, messageOnFalse__, __FILE__, __LINE__); \
      return Status_InvalidArgument;                                                                               \
    }                                                                                                              \
  } while (false)

#define GS_REQUIRE_OR_ABORT(requireCmd__, messageOnFalse__)                                                        \
  do {                                                                                                             \
    if (!(requireCmd__)) {                                                                                         \
      gsloge("Expression must be true [%s] - %s - at %s:%d", #requireCmd__, messageOnFalse__, __FILE__, __LINE__); \
      abort();                                                                                                     \
    }                                                                                                              \
  } while (false)

#define GS_REQUIRE_OR_RET_STATUS_FMT(requireCmd__, messageFmtOnFalse__, ...)                \
  do {                                                                                      \
    if (!(requireCmd__)) {                                                                  \
      gsloge("Expression must be true [%s] - at %s:%d", #requireCmd__, __FILE__, __LINE__); \
      gsloge(messageFmtOnFalse__, __VA_ARGS__);                                             \
      return Status_InvalidArgument;                                                        \
    }                                                                                       \
  } while (false)

#define GS_REQUIRE_OR_RET_RESULT_FMT(requireCmd__, messageFmtOnFalse__, ...)                \
  do {                                                                                      \
    if (!(requireCmd__)) {                                                                  \
      gsloge("Expression must be true [%s] - at %s:%d", #requireCmd__, __FILE__, __LINE__); \
      gsloge(messageFmtOnFalse__, __VA_ARGS__);                                             \
      return {.status = Status_InvalidArgument, .value = {}};                               \
    }                                                                                       \
  } while (false)

#define GS_REQUIRE_OR_RET_FMT(requireCmd__, returnOnFalse__, messageFmtOnFalse__, ...)      \
  do {                                                                                      \
    if (!(requireCmd__)) {                                                                  \
      gsloge("Expression must be true [%s] - at %s:%d", #requireCmd__, __FILE__, __LINE__); \
      gsloge(messageFmtOnFalse__, __VA_ARGS__);                                             \
      return returnOnFalse__;                                                               \
    }                                                                                       \
  } while (false)

#define GS_REQUIRE_OR_RET_RESULT(requireCmd__, messageOnFalse__)                                                   \
  do {                                                                                                             \
    if (!(requireCmd__)) {                                                                                         \
      gsloge("Expression must be true [%s] - %s - at %s:%d", #requireCmd__, messageOnFalse__, __FILE__, __LINE__); \
      return ERR_RESULT(Status_InvalidArgument);                                                                   \
    }                                                                                                              \
  } while (false)

#define GS_REQUIRE_OR_RET(requireCmd__, messageOnFalse__, retValueOnFalse__)                                       \
  do {                                                                                                             \
    if (!(requireCmd__)) {                                                                                         \
      gsloge("Expression must be true [%s] - %s - at %s:%d", #requireCmd__, messageOnFalse__, __FILE__, __LINE__); \
      return retValueOnFalse__;                                                                                    \
    }                                                                                                              \
  } while (false)

#define GS_REQUIRE_OR_THROW(requireCmd__, messageOnFalse__)                                                        \
  do {                                                                                                             \
    if (!(requireCmd__)) {                                                                                         \
      gsloge("Expression must be true [%s] - %s - at %s:%d", #requireCmd__, messageOnFalse__, __FILE__, __LINE__); \
      throw std::runtime_error("Failed assertion");                                                                \
    }                                                                                                              \
  } while (false)

#define GS_REQUIRE_OR_THROW_FMT(requireCmd__, messageFmtOnFalse__, ...)                     \
  do {                                                                                      \
    if (!(requireCmd__)) {                                                                  \
      gsloge("Expression must be true [%s] - at %s:%d", #requireCmd__, __FILE__, __LINE__); \
      gsloge(messageFmtOnFalse__, __VA_ARGS__);                                             \
      throw std::runtime_error("Failed assertion");                                         \
    }                                                                                       \
  } while (false)

#define GS_REQUIRE_EQ(requireExpectedVal__, requireCmd__, messageOnNotEq__)                                     \
  do {                                                                                                          \
    auto requireExpectedVal2__ = requireExpectedVal__;                                                          \
    auto requireActualVal__ = requireCmd__;                                                                     \
    if (requireExpectedVal2__ != requireActualVal__) {                                                          \
      GS_FAIL(                                                                                                  \
          messageOnNotEq__ << " - Expected [" << #requireCmd__ << "] to evaluate to [" << requireExpectedVal2__ \
                           << "] but it was [" << requireActualVal__ << ']');                                   \
    }                                                                                                           \
  } while (false)

#define GS_REQUIRE_LTE(requireCmd__, requireBound__, messageOnFail__)                                    \
  do {                                                                                                   \
    auto requireBound2__ = requireBound__;                                                               \
    auto requireActualVal__ = requireCmd__;                                                              \
    if (requireActualVal__ > requireBound2__) {                                                          \
      GS_FAIL(                                                                                           \
          messageOnFail__ << " - Expected [" << #requireCmd__ << "] (" << requireActualVal__ << ") <= [" \
                          << #requireBound__ << "] (" << requireBound2__ << ')');                        \
    }                                                                                                    \
  } while (false)

#define GS_REQUIRE_LT(requireCmd__, requireBound__, messageOnFail__)                                    \
  do {                                                                                                  \
    auto requireBound2__ = requireBound__;                                                              \
    auto requireActualVal__ = requireCmd__;                                                             \
    if (requireActualVal__ >= requireBound2__) {                                                        \
      GS_FAIL(                                                                                          \
          messageOnFail__ << " - Expected [" << #requireCmd__ << "] (" << requireActualVal__ << ") < [" \
                          << #requireBound__ << "] (" << requireBound2__ << ')');                       \
    }                                                                                                   \
  } while (false)

#define GS_REQUIRE_GTE(requireCmd__, requireBound__, messageOnFail__)                                    \
  do {                                                                                                   \
    auto requireBound2__ = requireBound__;                                                               \
    auto requireActualVal__ = requireCmd__;                                                              \
    if (requireActualVal__ < requireBound2__) {                                                          \
      GS_FAIL(                                                                                           \
          messageOnFail__ << " - Expected [" << #requireCmd__ << "] (" << requireActualVal__ << ") >= [" \
                          << #requireBound__ << "] (" << requireBound2__ << ')');                        \
    }                                                                                                    \
  } while (false)

#define GS_REQUIRE_GT(requireCmd__, requireBound__, messageOnFail__)                                    \
  do {                                                                                                  \
    auto requireBound2__ = requireBound__;                                                              \
    auto requireActualVal__ = requireCmd__;                                                             \
    if (requireActualVal__ <= requireBound2__) {                                                        \
      GS_FAIL(                                                                                          \
          messageOnFail__ << " - Expected [" << #requireCmd__ << "] (" << requireActualVal__ << ") > [" \
                          << #requireBound__ << "] (" << requireBound2__ << ')');                       \
    }                                                                                                   \
  } while (false)

/**
 * Must pass a block - surrounded with {}. Useful for defining noexcept functions
 */
#define CLEANUP_AND_ABORT_ON_EX(abortOnCatchBlock__, doOnCatch__) \
  {                                                               \
    try abortOnCatchBlock__ catch (const exception& e) {          \
      doOnCatch__;                                                \
      GS_FAIL("Exception [" << e.what() << "]");                  \
    }                                                             \
    catch (const std::string& str) {                              \
      doOnCatch__;                                                \
      GS_FAIL(str);                                               \
    }                                                             \
    catch (...) {                                                 \
      doOnCatch__;                                                \
      GS_FAIL("Unknown object caught");                           \
    }                                                             \
  }

#define ABORT_ON_EX(abortOnCatchBlock2__) CLEANUP_AND_ABORT_ON_EX(abortOnCatchBlock2__, static_cast<void>(0))

#endif  // GPUSDR_GSERRORS_H
