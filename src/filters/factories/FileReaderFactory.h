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

#ifndef GPUSDRPIPELINE_FILEREADERFACTORY_H
#define GPUSDRPIPELINE_FILEREADERFACTORY_H

#include <nlohmann/json.hpp>

#include "../FileReader.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class FileReaderFactory final : public IFileReaderFactory {
 public:
  explicit FileReaderFactory(IFactories* factories) noexcept
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept override {
    const auto obj = nlohmann::json::parse(jsonParameters);

    return ResultCast<Node>(FileReader::create(obj["filename"], mFactories));
  }

  Result<Source> createFileReader(const char* fileName) noexcept final {
    return FileReader::create(fileName, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;
  REF_COUNTED(FileReaderFactory);
};

#endif  // GPUSDRPIPELINE_FILEREADERFACTORY_H
