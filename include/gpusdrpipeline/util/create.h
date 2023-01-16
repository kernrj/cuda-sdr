//
// Created by Rick Kern on 1/14/23.
//

#ifndef GPUSDRPIPELINE_CREATE_H
#define GPUSDRPIPELINE_CREATE_H

#include <memory>
#include <stdexcept>

template <class FACTORY_TYPE, class... Args>
decltype(auto) create(FACTORY_TYPE factory, Args... args) {
  auto createdObject = factory->create(args...);

  if (createdObject == nullptr) {
    throw std::bad_alloc();
  }

  using ObjTypePtr = decltype(createdObject);
  using ObjType = typename std::remove_pointer<ObjTypePtr>::type;

  return std::shared_ptr<ObjType>(createdObject);
}

#endif  // GPUSDRPIPELINE_CREATE_H
