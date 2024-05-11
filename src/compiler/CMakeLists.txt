file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(compiler ${SOURCES})

# target_compile_options(compiler PRIVATE -Wall -Werror)

target_include_directories(compiler PUBLIC ../ SYSTEM ../../libs/)

target_link_libraries(compiler PUBLIC
    computationGraph executor tensor operations
    libutils)
