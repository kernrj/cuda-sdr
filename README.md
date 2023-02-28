This library enabled SDR processing on Nvidia GPUs.

This is in the tinkering phase, but bug reports and contributions are welcome. The interface is not yet stable.

To allocate classes, use the IFactories interface, which can be obtained from `getFactoriesSingleton()` in Factories.h.

This is a C++ library written to minimize ABI issues between compilers.
This leads to a few consequences:
- Exceptions aren't used across library/application boundaries. Instead, a `Result` struct is used to carry either a value or an error code.
- Only classes with virtual default-implementation destructors, default noexcept constructors, or pure virtual, noexcept methods are available.
- A ref-counting implementation is used instead of shared_ptr (delete/free occurs in the same code that created/allocated an object).
- All top-level functions use C linkage to avoid name-mangling issues between C++ compilers. They are also noexcept and return a Result struct if an error can occur.

The `Result` struct has a well-defined size and member offsets across compilers. Different C++ compilers may align struct members differently, so the alignment is explicit to allow for compiler ABI compatibility.
