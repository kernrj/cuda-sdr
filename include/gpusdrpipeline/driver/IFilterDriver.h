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

#ifndef GPUSDRPIPELINE_IFILTERDRIVER_H
#define GPUSDRPIPELINE_IFILTERDRIVER_H

#include <gpusdrpipeline/driver/IDriver.h>
#include <gpusdrpipeline/filters/Filter.h>

class IFilterDriver : public IDriver, public Filter {
 public:
  /**
   * In order to use this IFilterDriver as a Sink, set the Sink to delegate to.
   */
  virtual void setDriverInput(Sink* sink) noexcept = 0;

  /**
   * In order to use this IFilterDriver as a Source, set the Source to delegate to.
   */
  virtual void setDriverOutput(Source* source) noexcept = 0;

  ABSTRACT_IREF(IFilterDriver);
};

/*
 * The output needs to redirect readOutputs() to step the graph,
 * then write the driverOutput's output to the outputPorts.
 * readOutputs(buffers) {
 *   driverOutput->readOutputs(buffer)
 * }
 *
 * DriverOutputNode wraps the caller's buffers then does a doFilter()
 * setDriverOutput(sink) connects that sink to the DriverOutputNode wrapper.
 *
 * When driverInput->readOutput() is called, it should provide whatever data
 * it received as input via IFilterDriver.request/commitBuffer().
 *
 * IFD.request() finds the first Sink connected to it and forwards the request for
 * buffers to it.
 *
 * The target source is set as a member var each time IFD::requestBuffer() is called.
 * The driverInput's Sink methods are delegated to it.
 *
 * If connection changes are made at run-time, it's up to the implementation to make sure
 *   data isn't buffered in the old target source before a new one is set/old one removed.
 * It will be null when driverInput is not connected (e.g. Source that reads from a file).
 *
 *
 */

#endif  // GPUSDRPIPELINE_IFILTERDRIVER_H
