//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_IBUFFERRANGE_H
#define GPUSDR_IBUFFERRANGE_H

#include <cstddef>
#include <stdexcept>
#include <string>

class IBufferRange {
 public:
  virtual ~IBufferRange() = default;

  /**
   * Returns the number of bytes available from an IBuffer's base().
   */
  [[nodiscard]] virtual size_t capacity() const = 0;

  /**
   * Returns the index of the first used byte in a buffer.
   */
  [[nodiscard]] virtual size_t offset() const = 0;

  /**
   * Returns the exclusive upper bound of the used data in a buffer.
   */
  [[nodiscard]] virtual size_t endOffset() const = 0;

  /**
   * Sets the range of used bytes: offset and endOffset.
   * offset must be <= endOffset.
   *
   * @param offset
   * @param endOffset
   */
  virtual void setUsedRange(size_t offset, size_t endOffset) = 0;

  [[nodiscard]] virtual size_t used() const { return endOffset() - offset(); }
  [[nodiscard]] virtual size_t remaining() const { return capacity() - endOffset(); }
  [[nodiscard]] virtual bool hasRemaining() const { return remaining() > 0; }
  virtual void clearRange() { setUsedRange(0, 0); }

  virtual void increaseOffset(size_t increaseBy) {
    size_t newStartOffset = offset() + increaseBy;
    if (newStartOffset > endOffset()) {
      throw std::runtime_error(
          "New start offset [" + std::to_string(newStartOffset) + "] exceeds the end offset ["
          + std::to_string(endOffset()) + "]");
    }

    setUsedRange(newStartOffset, endOffset());
  }

  virtual void increaseEndOffset(size_t increaseBy) {
    size_t newEndOffset = endOffset() + increaseBy;
    if (newEndOffset > capacity()) {
      throw std::runtime_error(
          "New end offset [" + std::to_string(newEndOffset) + "] exceeds the capacity [" + std::to_string(endOffset())
          + "]");
    }
    setUsedRange(offset(), newEndOffset);
  }
};

#endif  // GPUSDR_IBUFFERRANGE_H
