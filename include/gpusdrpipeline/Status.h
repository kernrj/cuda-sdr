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

#ifndef GPUSDRPIPELINE_STATUS_H
#define GPUSDRPIPELINE_STATUS_H

#include <stdexcept>

using Status = uint32_t;
enum Status_ {
  Status_Success,
  Status_UnknownError,
  Status_OutOfMemory,
  Status_RuntimeError,
  Status_InvalidArgument,
  Status_InvalidState,
  Status_OutOfRange,
  Status_TimedOut,
  Status_NotFound,
  Status_ParseError,
};

/**
 * When the status is an error, an exception is thrown.
 *
 * This should not be used to throw exceptions between application and library code!
 * Its purpose is to handle a status value passed between application and library code.
 * @param status
 */
inline void throwIfError(Status status) {
  switch (status) {
    case Status_Success:
      return;

    case Status_UnknownError:
      throw std::runtime_error("Unknown Error");

    case Status_OutOfMemory:
      throw std::bad_alloc();

    case Status_RuntimeError:
      throw std::runtime_error("Error");

    case Status_InvalidArgument:
      throw std::invalid_argument("Invalid Argument");

    case Status_InvalidState:
      throw std::runtime_error("Invalid State");

    case Status_OutOfRange:
      throw std::out_of_range("Out of Range");

    case Status_TimedOut:
      throw std::runtime_error("Timed Out");

    case Status_NotFound:
      throw std::runtime_error("Not Found");

    case Status_ParseError:
      throw std::runtime_error("Parse Error");

    default:
      throw std::runtime_error("Error type [" + std::to_string(status) + "]");
  }
}

#endif  // GPUSDRPIPELINE_STATUS_H
