set(BINARY ${PROJECT_NAME}_test)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES true *.h *.cpp)

# set(TEST_SOURCES main.cpp gpuTests.cpp functionsTests.cpp)

# set(TEST_SOURCES main.cpp functionsTests.cpp graphTests.cpp tensorTests.cpp executorTest.cpp)

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

add_executable(${BINARY} ${TEST_SOURCES}) # ${HEADERS} ${CPPERS})

gtest_discover_tests(${BINARY})

target_include_directories(${BINARY} PUBLIC ../src ../libs/)

target_link_libraries(${BINARY} gtest gmock 
    ${EXTERNAL_LIBRARIES}
    ${INTERNAL_LIBRARIES})