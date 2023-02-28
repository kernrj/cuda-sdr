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
#ifndef GPUSDRPIPELINE_IREF_H
#define GPUSDRPIPELINE_IREF_H

#include <gpusdrpipeline/GSDefs.h>

#include <atomic>
#include <cstdio>

/**
 * IRef is the base interface for reference-counted objects.
 *
 * Implementing classes should use virtual inheritance when extending IRef. Otherwise, ImmutableRef and Ref won't
 * compile.
 */
class IRef {
 public:
  virtual void ref() const noexcept = 0;
  virtual void unref() const noexcept = 0;

 protected:
  IRef() noexcept = default;
  virtual ~IRef() = default;
};

/**
 * An unmodifiable reference to an IRef. When a raw pointer is needed, a ConstRef needs to be used, because the IRef
 * can be swapped in a Ref and cause use of a deleted object.
 */
template <typename T>
class ImmutableRef final {
 public:
  ImmutableRef() noexcept
      : mReffed(nullptr) {}

  ImmutableRef(T* reffed) noexcept
      : mReffed(reffed) {
    if (mReffed != nullptr) {
      mReffed->ref();
    }
  }

  ImmutableRef(const ImmutableRef<T>& ref) noexcept
      : mReffed(ref.mReffed) {
    if (mReffed != nullptr) {
      mReffed->ref();
    }
  }

  ImmutableRef(ImmutableRef<T>&& ref) noexcept
      : mReffed(ref.mReffed) {
    if (mReffed != nullptr) {
      mReffed->ref();
    }
  }

  ~ImmutableRef() noexcept {
    if (mReffed != nullptr) {
      mReffed->unref();
    }
  }

  ImmutableRef& operator=(const ImmutableRef<T>& ref) = delete;
  ImmutableRef& operator=(ImmutableRef<T>&& ref) = delete;

  operator T*() const noexcept { return mReffed; }

  T* operator->() const noexcept { return mReffed; }
  T* get() const noexcept { return mReffed; }

  bool operator==(const ImmutableRef<T>& other) const noexcept { return mReffed == other.mReffed; }
  bool operator!=(const ImmutableRef<T>& other) const noexcept { return mReffed != other.mReffed; }
  bool operator==(T* other) const noexcept { return mReffed == other; }
  bool operator!=(T* other) const noexcept { return mReffed != other; }

 private:
  T* const mReffed;
};

template <typename T>
using ConstRef = const ImmutableRef<T>;

template <typename T, typename = typename std::enable_if<std::is_base_of<IRef, T>::value>::type>
class Ref final {
 public:
  Ref() noexcept
      : mReffed(nullptr) {}

  Ref(T* reffed) noexcept
      : mReffed(nullptr) {
    reset(reffed);
  }

  Ref(const Ref& ref) noexcept
      : mReffed(nullptr) {
    reset(ref.mReffed.load());
  }

  Ref(Ref&& ref) noexcept
      : mReffed(nullptr) {
    reset(ref.mReffed);
    ref.mReffed.store(nullptr);
  }

  Ref(const ImmutableRef<T>& other) noexcept
      : mReffed(nullptr) {
    reset(other);
  }

  ~Ref() noexcept {
    if (mReffed != nullptr) {
      mReffed.load()->unref();
    }
  }

  Ref& operator=(T* reffed) noexcept {
    reset(reffed);
    return *this;
  }

  Ref& operator=(const Ref& ref) noexcept {
    if (&ref == this) {
      return *this;
    }

    reset(ref.get());
    return *this;
  }

  Ref& operator=(const ImmutableRef<T>& other) noexcept {
    reset(other);
    return *this;
  }

  Ref& operator=(Ref&& ref) noexcept {
    if (&ref == this) {
      return *this;
    }

    reset(ref.get());
    ref.reset();

    return *this;
  }

  void reset() noexcept { reset(nullptr); }

  void reset(T* reffed) noexcept {
    if (reffed != nullptr) {
      /*
       * IREF_CAST_CLASS is usually IRef, but can be different for cases where multiple IRefs are in the inheritance
       * tree.
       */
      reffed->ref();
    }

    const T* oldReffed = atomic_exchange_explicit(&mReffed, reffed, std::memory_order_seq_cst);

    if (oldReffed != nullptr) {
      oldReffed->unref();  // unref() after mReffed->ref() in case this->mReffed == reffed.
    }
  }

  operator ImmutableRef<T>() const noexcept { return ImmutableRef<T>(mReffed.load()); }
  ImmutableRef<T> operator->() const noexcept { return ImmutableRef<T>(mReffed.load()); }

  /**
   * Returns a ImmutableRef for the IRef.
   * It doesn't return the raw pointer because it could be swapped after return, but before the caller refs it.
   * ImmutableRef prevents this issue because it refs before return, and only unrefs once it goes out of scope.
   */
  ImmutableRef<T> get() const noexcept { return ImmutableRef<T>(mReffed.load()); }

  bool operator==(const Ref<T>& other) const noexcept { return mReffed.load() == other.mReffed.load(); }
  bool operator!=(const Ref<T>& other) const noexcept { return mReffed.load() != other.mReffed.load(); }
  bool operator==(T* other) const noexcept { return mReffed.load() == other; }
  bool operator!=(T* other) const noexcept { return mReffed.load() != other; }

 private:
  std::atomic<T*> mReffed;
};

/**
 * Classes can delegate to RefCt for their implementation of IRef.
 */
template <class T>
class RefCt final {
 public:
  /**
   * onRefCountZero(refContext) is called once the ref-count reaches 0.
   *
   * For example:
   * ------------
   *
   * class MyClass final : public virtual IRef {
   * public:
   *   MyClass() : mRef(this, deleteThis) {}
   *
   * private:
   *   RefCt<MyClass> mRef;
   *
   * private:
   *   ~MyClass() final = default;
   *   static void deleteThis(MyClass* myClass) { delete myClass; }
   *   void ref() noexcept final { mRef.ref(); }
   *   void unref() noexcept final { mRef.unref(); }
   * };
   */
  RefCt(T* refContext, void (*onRefCountZero)(T* refContext) noexcept) noexcept
      : mRefContext(refContext),
        mOnRefCountZero(onRefCountZero) {}

  RefCt(const RefCt<T>&) = delete;
  RefCt(RefCt<T>&&) = delete;
  RefCt& operator=(const RefCt&) = delete;
  RefCt& operator=(RefCt&&) = delete;

  ~RefCt() = default;

  void ref() const noexcept { mRefCount.fetch_add(1); }

  void unref() const noexcept {
    size_t previousRefCount;

    do {
      previousRefCount = mRefCount.load();
      if (previousRefCount == 0) {
        break;  // never reffed
      }
    } while (!mRefCount.compare_exchange_strong(previousRefCount, previousRefCount - 1));

    if (previousRefCount <= 1) {
      mOnRefCountZero(mRefContext);
    }
  }

 private:
  mutable std::atomic_size_t mRefCount {0};
  T* const mRefContext;
  void (*const mOnRefCountZero)(T* refContext) noexcept;
};

#endif  // GPUSDRPIPELINE_IREF_H
