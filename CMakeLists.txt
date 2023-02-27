cmake_minimum_required(VERSION 3.19)

set(LIB_PROJECT_NAME gpusdrpipeline)
set(LIB_TARGET_NAME ${LIB_PROJECT_NAME})

function(proj_set_default_warning_flags TARGET SCOPE)
    if (MSVC)
        target_compile_option(${TARGET} ${SCOPE} /W4)
    else ()
        set(
            PROJ_NATIVE_COMPILER_WARNING_FLAGS
            -Wall
            -Wextra
            -Werror=unused-result
            -Werror=return-type
            -Werror=suggest-override
            -Werror=format
            -Wno-return-type-c-linkage
        )

        list(JOIN PROJ_NATIVE_COMPILER_WARNING_FLAGS "," PROJ_CUDA_COMPILER_WARNING_FLAGS)
        set(PROJ_CUDA_COMPILER_WARNING_FLAGS "--compiler-options ${PROJ_CUDA_COMPILER_WARNING_FLAGS}")

        message("Setting target warning properties [${TARGET}] scope [${SCOPE}]")
        message("Native compiler '${PROJ_NATIVE_COMPILER_WARNING_FLAGS}'")
        message("CUDA compiler '${PROJ_CUDA_COMPILER_WARNING_FLAGS}'")
        target_compile_options(
            ${TARGET}
            ${SCOPE}
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${PROJ_CUDA_COMPILER_WARNING_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:${PROJ_NATIVE_COMPILER_WARNING_FLAGS}>
            $<$<COMPILE_LANGUAGE:C>:${PROJ_NATIVE_COMPILER_WARNING_FLAGS}>
        )
    endif ()
endfunction()

set(CMAKE_CXX_COMPILER /usr/bin/clang++)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_C_COMPILER /usr/bin/clang)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(PROJ_CUDA_ARCHITECTURES "75")

set(VERBOSE 1)

include_directories(
    SYSTEM
    /usr/include/c++/11
    /usr/include/x86_64-linux-gnu/c++/11
)

project(${LIB_PROJECT_NAME})

set(LIB_PROJECT_ROOT_DIR "${PROJECT_SOURCE_DIR}")

#enable_language(CUDA)
#include(CheckLanguage)
#check_language(CUDA)

#if (NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 17)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif ()

add_subdirectory(src)

option(USE_TESTS "Enable unit testing" ON)
if (USE_TESTS)
    message("Tests enabled.")
    enable_testing()
    add_subdirectory(tests)
endif ()
