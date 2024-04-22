cmake_minimum_required(VERSION 3.21)
project(NSTTF)

include(clang-format.cmake)

set(BINARY ${PROJECT_NAME}_main)

enable_testing()

set(EXTERNAL_LIBRARIES
        libclew
        libgpu
        libutils
        CACHE INTERNAL "List of external libraries")

set(INTERNAL_LIBRARIES
        cl_funcs 
        computationGraph
        operations
        tensor
        utils
        
        CACHE INTERNAL "list of internal libraries")

add_subdirectory(libs)
add_subdirectory(tests)
add_subdirectory(src)

set(CMAKE_CXX_STANDARD 20)


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

# add_compile_definitions(_DISABLE_VECTOR_ANNOTATION)
# add_compile_definitions(_DISABLE_STRING_ANNOTATION)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами
# find_package(OpenMP)

# if(OpenMP_CXX_FOUND)
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# else()
#         message(WARNING "OpenMP not found!")
# endif()


# file(GLOB_RECURSE
#      ALL_CXX_SOURCE_FILES
#      *.[chi]pp *.[chi]xx *.cc *.hh *.ii *.[CHI]
#      )

# find_program(CLANG_FORMAT "clang-format")
# if(CLANG_FORMAT)
#     add_custom_target(
#         clang-format
#         COMMAND /usr/bin/clang-format
#         -i
#         -style=file
#         ${ALL_CXX_SOURCE_FILES}
#         )
# else()
#     message(WARNING "clang-format not found!")
# endif()


add_executable(${BINARY} src/main.cpp)

target_include_directories(${BINARY} PUBLIC src libs)

target_link_libraries(${BINARY}
        ${EXTERNAL_LIBRARIES}
        ${INTERNAL_LIBRARIES}
)