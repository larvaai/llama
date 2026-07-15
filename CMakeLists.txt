cmake_minimum_required(VERSION 3.24)
project(model_worker_native VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LLVM-MinGW links its C++ and OpenMP runtimes dynamically on Windows.  Keep
# those DLLs beside the executables so they can run outside the compiler shell.
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  get_filename_component(LLVM_MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)

  function(copy_llvm_mingw_runtimes target)
    foreach(runtime_dll IN ITEMS libc++.dll libunwind.dll libomp.dll libwinpthread-1.dll)
      if(EXISTS "${LLVM_MINGW_BIN_DIR}/${runtime_dll}")
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                  "${LLVM_MINGW_BIN_DIR}/${runtime_dll}"
                  "$<TARGET_FILE_DIR:${target}>/${runtime_dll}"
          VERBATIM
        )
      endif()
    endforeach()
  endfunction()
endif()

set(LLAMA_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/controlled_inference/vendor/llama.cpp" CACHE PATH "Pinned llama.cpp b10012 source")
if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/include/llama.h")
  message(FATAL_ERROR "Pinned llama.cpp headers not found at ${LLAMA_CPP_SOURCE_DIR}; initialize the b10012 dependency")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build pinned llama.cpp common runtime" FORCE)
option(
  AGENT_HARNESS_LLGUIDANCE
  "Build optional llguidance constrained decoding support (requires Rust cargo)"
  OFF
)
if(AGENT_HARNESS_LLGUIDANCE)
  find_program(CARGO_EXECUTABLE cargo)
  if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR
      "AGENT_HARNESS_LLGUIDANCE=ON requires Rust cargo. Install Rust or configure "
      "with -DAGENT_HARNESS_LLGUIDANCE=OFF to use llama.cpp's JSON grammar fallback."
    )
  endif()
endif()
set(
  LLAMA_LLGUIDANCE
  ${AGENT_HARNESS_LLGUIDANCE}
  CACHE BOOL "Use pinned llguidance constrained decoding"
  FORCE
)
add_subdirectory("${LLAMA_CPP_SOURCE_DIR}" llama.cpp EXCLUDE_FROM_ALL)

# llama.cpp b10012 uses UTF-8 literals with APIs that still accept const char *.
# Scope the compatibility flag to the pinned library: applying it globally
# breaks nlohmann/json's C++20 filesystem conversions in our executables.
if(MSVC)
  target_compile_options(llama PRIVATE /Zc:char8_t-)
endif()

add_executable(model-worker-native
  native/model_worker_main.cpp
  native/generation_safety.cpp
  native/pending_cancel_registry.cpp
  native/reasoning_phase_controller.cpp
  native/sequence_engine.cpp
  native/ipc_protocol.cpp
)
target_include_directories(model-worker-native PRIVATE native "${LLAMA_CPP_SOURCE_DIR}/vendor")
target_link_libraries(model-worker-native PRIVATE llama)
if(COMMAND copy_llvm_mingw_runtimes)
  copy_llvm_mingw_runtimes(model-worker-native)
endif()

add_executable(inference-runtime-native
  native/inference_runtime_main.cpp
  native/generation_safety.cpp
  native/reasoning_phase_controller.cpp
  native/sequence_engine.cpp
)
target_include_directories(inference-runtime-native PRIVATE native "${LLAMA_CPP_SOURCE_DIR}/vendor" "${LLAMA_CPP_SOURCE_DIR}/common")
target_link_libraries(inference-runtime-native PRIVATE llama llama-common)
if(COMMAND copy_llvm_mingw_runtimes)
  copy_llvm_mingw_runtimes(inference-runtime-native)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(model-worker-native-tests
    native/tests/test_reasoning_phase_controller.cpp
    native/generation_safety.cpp
    native/pending_cancel_registry.cpp
    native/reasoning_phase_controller.cpp
    native/sequence_engine.cpp
    native/ipc_protocol.cpp
  )
  target_include_directories(model-worker-native-tests PRIVATE native)
  if(COMMAND copy_llvm_mingw_runtimes)
    copy_llvm_mingw_runtimes(model-worker-native-tests)
  endif()
  add_test(NAME model-worker-native-tests COMMAND model-worker-native-tests)
endif()
