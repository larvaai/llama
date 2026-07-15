cmake_minimum_required(VERSION 3.24)
project(model_worker_native VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLAMA_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/controlled_inference/vendor/llama.cpp" CACHE PATH "Pinned llama.cpp b10012 source")
if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/include/llama.h")
  message(FATAL_ERROR "Pinned llama.cpp headers not found at ${LLAMA_CPP_SOURCE_DIR}; initialize the b10012 dependency")
endif()

add_subdirectory("${LLAMA_CPP_SOURCE_DIR}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(model-worker-native
  native/model_worker_main.cpp
  native/reasoning_phase_controller.cpp
  native/ipc_protocol.cpp
)
target_include_directories(model-worker-native PRIVATE native "${LLAMA_CPP_SOURCE_DIR}/vendor")
target_link_libraries(model-worker-native PRIVATE llama)

include(CTest)
if(BUILD_TESTING)
  add_executable(model-worker-native-tests
    native/tests/test_reasoning_phase_controller.cpp
    native/reasoning_phase_controller.cpp
    native/ipc_protocol.cpp
  )
  target_include_directories(model-worker-native-tests PRIVATE native)
  add_test(NAME model-worker-native-tests COMMAND model-worker-native-tests)
endif()
