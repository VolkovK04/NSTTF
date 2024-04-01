set(BINARY ${CMAKE_PROJECT_NAME}_test)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

set(SOURCES ${TEST_SOURCES})

include_directories(libs)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(GTest)

# add_executable(${BINARY} ${TEST_SOURCES}
# target_include_directories(${BINARY} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(
  # ../src/cl_build_headers/
  # ../src/compputationGraph/
  # ../src/operations/
  # ../src/tensor/
  # ../src/utils/
  ../src
  )

set(HEADERS 
  ../src/computationGraph/computationGraph.h
  ../src/computationGraph/node.h

  ../src/operations/abstractOperation.h
  ../src/tensor/tensor.h

  ../src/utils/utils.h
  ../src/utils/compiler.h
  ../src/utils/functions.h
  ../src/utils/instruction.h
  ../src/utils/graphExecutor.h
)

set(CPPERS
  ../src/computationGraph/graph.cpp
  ../src/computationGraph/node.cpp

  ../src/operations/abstractOperation.cpp
  ../src/tensor/tensor.cpp
  ../src/tensor/gpuPointer.cpp
  ../src/tensor/ramPointer.cpp

  ../src/utils/utils.cpp
  ../src/utils/compiler.cpp
  ../src/utils/functions.cpp
  ../src/utils/graphExecutor.cpp
)
  
  
add_executable(${BINARY} ${TEST_SOURCES} ${HEADERS} ${CPPERS})

gtest_add_tests(TARGET ${BINARY})

target_link_libraries(${BINARY} gtest gmock libclew libgpu libutils)