set(TEST_BINARY ${PROJECT_NAME}_test)
set(EFFICIENCY_TEST_BINARY ${PROJECT_NAME}_efficiency_test)

set(TEST_SOURCES 
  main.cpp 
  gpuTests.cpp 
  functionsTests.cpp 
  graphTests.cpp 
  tensorTests.cpp 
  executorTest.cpp
  functionalTests.cpp
  derivativeTests.cpp
)

set(EFFICIENCY_TEST_SOURCES
  efficiencyTest.cpp
)

# include_directories(libs)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(GTest)

add_executable(${TEST_BINARY} ${TEST_SOURCES})
add_executable(${EFFICIENCY_TEST_BINARY} ${EFFICIENCY_TEST_SOURCES})

gtest_discover_tests(${TEST_BINARY})
gtest_discover_tests(${EFFICIENCY_TEST_BINARY})

# target_include_directories(${TEST_BINARY} PUBLIC ../src ../libs/)

target_link_libraries(${TEST_BINARY} gtest gmock 
    ${EXTERNAL_LIBRARIES}
    ${INTERNAL_LIBRARIES})

target_link_libraries(${EFFICIENCY_TEST_BINARY} gtest gmock 
    ${EXTERNAL_LIBRARIES}
    ${INTERNAL_LIBRARIES})