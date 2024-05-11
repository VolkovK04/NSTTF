cmake_minimum_required(VERSION 3.21)
project(NSTTF)

set(CMAKE_CXX_STANDARD 23)
include(clang-format.cmake)

set(BINARY ${PROJECT_NAME}_main)

add_compile_definitions(_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")


enable_testing()

set(EXTERNAL_LIBRARIES
        libclew
        libgpu
        libutils
        CACHE INTERNAL "List of external libraries")

set(INTERNAL_LIBRARIES
        computationGraph
        operations
        tensor
        compiler
        neuralNetwork
        CACHE INTERNAL "list of internal libraries")

include_directories(src, libs)

add_subdirectory(libs)
add_subdirectory(tests)
add_subdirectory(src)


add_executable(${BINARY} src/main.cpp)

target_include_directories(${BINARY} PUBLIC src libs)

target_link_libraries(${BINARY}
        ${EXTERNAL_LIBRARIES}
        ${INTERNAL_LIBRARIES}
)

if(CLANG_FORMAT)
        add_dependencies(${BINARY} clang-format) # run format before build main
endif()