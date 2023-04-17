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

#ifndef GPUSDRPIPELINE_IEXECDEVICE_H
#define GPUSDRPIPELINE_IEXECDEVICE_H

#include <gpusdrpipeline/IRef.h>
#include <gpusdrpipeline/buffers/IAllocatorFactory.h>

class IExecDevice : public virtual IRef {
 public:
  virtual Result<IAllocatorFactory> getAllocatorFactory() = 0;
  virtual Result<void> getNativeObject() = 0;

  ABSTRACT_IREF(IExecDevice);
};

class IExecDeviceList : public virtual IRef {
 public:
  virtual Result<IExecDevice> getDevice(const char* deviceId) = 0;
  virtual Result<size_t> getDeviceCount() = 0;
  virtual Result<size_t> getDeviceId(size_t deviceIndex, char* deviceId, size_t deviceIdBufferLength) = 0;

  ABSTRACT_IREF(IExecDeviceList);
};

#endif  // GPUSDRPIPELINE_IEXECDEVICE_H
