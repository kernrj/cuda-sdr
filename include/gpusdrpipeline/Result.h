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

#ifndef GPUSDRPIPELINE_RESULT_H
#define GPUSDRPIPELINE_RESULT_H

#include <gpusdrpipeline/GSLog.h>
#include <gpusdrpipeline/IRef.h>
#include <gpusdrpipeline/Status.h>

/**
 * A result that contains an IRef sub-class.
 * Use takeResult() to correctly unwrap a RefResult without memory leaks.
 */
template <typename T>
#pragma pack(push, 8)
struct RefResult {
  using ValueType = T*;

  const Status status;
  T* const value;
};
#pragma pack(pop)

/**
 * A result containing a (non-IRef) value, like an int or char.
 */
template <typename T>
#pragma pack(push, 8)
struct ValResult {
  using ValueType = T;

  Status status;
  T value;
};
#pragma pack(pop)

template <typename T>
using Result = typename std::conditional<std::is_base_of<IRef, T>::value, RefResult<T>, ValResult<T>>::type;

/**
 * Returns the value from the result, or throws an exception on error (unreffing the result).
 */
template <typename T>
ImmutableRef<T> unwrap(RefResult<T>* result) {
  ImmutableRef<T> value = result->value;
  result->value = nullptr;

  throwIfError(result->status);

  return value;
}

template <typename T>
ImmutableRef<T> unwrap(RefResult<T>&& result) {
  ImmutableRef<T> value = result.value;

  throwIfError(result.status);

  return value;
}

/**
 * Returns the value from the result, or throws an exception.
 */
template <typename T>
T unwrap(ValResult<T>* result) {
  throwIfError(result->status);

  return result->value;
}

template <typename T>
T unwrap(ValResult<T>&& result) {
  throwIfError(result.status);

  return result.value;
}

#define UNWRAP_OR_FWD_RESULT(assignValueToVar__, unwrapCmd__)                    \
  do {                                                                           \
    auto result__ = unwrapCmd__;                                                 \
    if (result__.status != Status_Success) {                                     \
      gsloge("Error in result [%s] at %s:%d", #unwrapCmd__, __FILE__, __LINE__); \
      return {                                                                   \
          .status = result__.status,                                             \
          .value = {},                                                           \
      };                                                                         \
    }                                                                            \
                                                                                 \
    assignValueToVar__ = result__.value;                                         \
  } while (false)

#define WARN_IF_ERR(cmdReturningStatus__)                          \
  do {                                                             \
    const Status status__ = cmdReturningStatus__;                  \
    if (status__ != Status_Success) {                              \
      gsloge("Error [%d] at %s:%d", status__, __FILE__, __LINE__); \
    }                                                              \
  } while (false)

#define FWD_IF_ERR(cmdReturningStatus__)                           \
  do {                                                             \
    const Status status__ = cmdReturningStatus__;                  \
    if (status__ != Status_Success) {                              \
      gsloge("Error [%d] at %s:%d", status__, __FILE__, __LINE__); \
      return status__;                                             \
    }                                                              \
  } while (false)

#define THROW_IF_ERR(cmdReturningStatus__)                         \
  do {                                                             \
    const Status status__ = cmdReturningStatus__;                  \
    if (status__ != Status_Success) {                              \
      gsloge("Error [%d] at %s:%d", status__, __FILE__, __LINE__); \
      throwIfError(status__);                                      \
    }                                                              \
  } while (false)

#define RET_IF_ERR(cmdReturningStatus__, returnValueOnErr__)       \
  do {                                                             \
    const Status status__ = cmdReturningStatus__;                  \
    if (status__ != Status_Success) {                              \
      gsloge("Error [%d] at %s:%d", status__, __FILE__, __LINE__); \
      return returnValueOnErr__;                                   \
    }                                                              \
  } while (false)

#define FWD_IN_RESULT_IF_ERR(cmdReturningStatus__)                 \
  do {                                                             \
    const Status status__ = cmdReturningStatus__;                  \
    if (status__ != Status_Success) {                              \
      gsloge("Error [%d] at %s:%d", status__, __FILE__, __LINE__); \
      return {.status = status__, .value = {}};                    \
    }                                                              \
  } while (false)

#define UNWRAP_OR_FWD_STATUS(assignValueToVar__, unwrapCmd__)                    \
  do {                                                                           \
    auto result__ = unwrapCmd__;                                                 \
    if (result__.status != Status_Success) {                                     \
      gsloge("Error in result [%s] at %s:%d", #unwrapCmd__, __FILE__, __LINE__); \
      return result__.status;                                                    \
    }                                                                            \
                                                                                 \
    assignValueToVar__ = result__.value;                                         \
  } while (false)

template <class T>
GS_FMT_ATTR(2, 3)
inline bool printIfError(Result<T>&& result, GS_FMT_STR(const char* fmt), ...) noexcept {
  if (result.status != Status_Success) {
    va_list args;
    va_start(args, fmt);
    gsvlog(GSLOG_ERROR, fmt, args);
    va_end(args);
  }

  return std::move(result.value);
}

#define UNWRAP_OR_RETURN(assignValueToVar__, unwrapCmd__, retOnError__)                                \
  do {                                                                                                 \
    auto result__ = unwrapCmd__;                                                                       \
    if (result__.status != Status_Success) {                                                           \
      gsloge("Error [%d] in result [%s] at %s:%d", result__.status, #unwrapCmd__, __FILE__, __LINE__); \
      return retOnError__;                                                                             \
    }                                                                                                  \
    assignValueToVar__ = result__.value;                                                               \
  } while (false)

#define DO_OR_FWD_ERR(unwrapCmd__)                                               \
  do {                                                                           \
    auto result__ = unwrapCmd__;                                                 \
    if (result__.status != Status_Success) {                                     \
      gsloge("Error in result [%s] at %s:%d", #unwrapCmd__, __FILE__, __LINE__); \
      return {                                                                   \
          .status = result__.status,                                             \
          .value = {},                                                           \
      };                                                                         \
    }                                                                            \
  } while (false)

#define NON_NULL_OR_RET(ptr__)                                            \
  do {                                                                    \
    auto ptrVar__ = ptr__;                                                \
    if (ptrVar__ == nullptr) {                                            \
      gsloge("%s cannot be null - at %s:%d", #ptr__, __FILE__, __LINE__); \
      return ERR_RESULT(Status_OutOfMemory);                              \
    }                                                                     \
  } while (false)

#define NON_NULL_PARAM_OR_RET(ptr__)                                      \
  do {                                                                    \
    auto ptrVar__ = ptr__;                                                \
    if (ptrVar__ == nullptr) {                                            \
      gsloge("%s cannot be null - at %s:%d", #ptr__, __FILE__, __LINE__); \
      return ERR_RESULT(Status_InvalidArgument);                          \
    }                                                                     \
  } while (false)

#define RET_PTR_OR_ERR(cmd__, ptr__)                                      \
  do {                                                                    \
    auto ptrVal__ = ptr__;                                                \
    if (ptrVar__ == nullptr) {                                            \
      gsloge("%s cannot be null - at %s:%d", #ptr__, __FILE__, __LINE__); \
      return ERR_RESULT(Status_OutOfMemory);                              \
    }                                                                     \
                                                                          \
    return {                                                              \
        .status = Status_Success,                                         \
        .value = ptrVal__,                                                \
    };                                                                    \
  } while (false)

#define DO_OR_RET_ERR_RESULT(doCmd__)                \
  do {                                               \
    try {                                            \
      doCmd__;                                       \
    } catch (const std::bad_alloc& stlEx__) {        \
      return ERR_RESULT(Status_OutOfMemory);         \
    } catch (const std::out_of_range& stlEx__) {     \
      return ERR_RESULT(Status_OutOfRange);          \
    } catch (const std::invalid_argument& stlEx__) { \
      return ERR_RESULT(Status_InvalidArgument);     \
    } catch (const std::runtime_error& stlEx__) {    \
      return ERR_RESULT(Status_RuntimeError);        \
    } catch (...) {                                  \
      return ERR_RESULT(Status_UnknownError);        \
    }                                                \
  } while (false)

#define IF_CATCH_RETURN_STATUS                   \
  catch (const std::bad_alloc& stlEx__) {        \
    return Status_OutOfMemory;                   \
  }                                              \
  catch (const std::out_of_range& stlEx__) {     \
    return Status_OutOfRange;                    \
  }                                              \
  catch (const std::invalid_argument& stlEx__) { \
    return Status_InvalidArgument;               \
  }                                              \
  catch (const std::runtime_error& stlEx__) {    \
    return Status_RuntimeError;                  \
  }                                              \
  catch (...) {                                  \
    return Status_UnknownError;                  \
  }

#define IF_CATCH_RETURN_RESULT                   \
  catch (const std::bad_alloc& stlEx__) {        \
    return ERR_RESULT(Status_OutOfMemory);       \
  }                                              \
  catch (const std::out_of_range& stlEx__) {     \
    return ERR_RESULT(Status_OutOfRange);        \
  }                                              \
  catch (const std::invalid_argument& stlEx__) { \
    return ERR_RESULT(Status_InvalidArgument);   \
  }                                              \
  catch (const std::runtime_error& stlEx__) {    \
    return ERR_RESULT(Status_RuntimeError);      \
  }                                              \
  catch (...) {                                  \
    return ERR_RESULT(Status_UnknownError);      \
  }

#define DO_OR_RET_STATUS(doCmd__) \
  do {                            \
    try {                         \
      doCmd__;                    \
    }                             \
    IF_CATCH_RETURN_STATUS        \
  } while (false)

template <typename T>
RefResult<T> makeRefResultNonNull(T* reffed) noexcept {
  return {
      .status = reffed != nullptr ? Status_Success : Status_OutOfMemory,
      .value = reffed,
  };
}

template <typename T>
RefResult<T> makeRefResultNonNull(const ImmutableRef<T>& reffed) noexcept {
  return {
      .status = reffed != nullptr ? Status_Success : Status_OutOfMemory,
      .value = reffed,
  };
}

template <typename T>
Result<T> makeRefResultNullable(T* reffed) noexcept {
  return {
      .status = Status_Success,
      .value = reffed,
  };
}

template <typename T>
Result<T> makeValResult(T value) noexcept {
  return {
      .status = Status_Success,
      .value = value,
  };
}

#define ERR_RESULT(errResultStatus__)        \
  {                                          \
    .status = errResultStatus__, .value = {} \
  }

template <typename T>
Result<T> errResult(Status status) {
  return {
      .status = status,
      .value = {},
  };
}

#endif  // GPUSDRPIPELINE_RESULT_H
