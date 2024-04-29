file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(compiler ${SOURCES})

target_include_directories(compiler PUBLIC ../ ../../libs/)

target_link_libraries(compiler PUBLIC
    computationGraph executor tensor operations
    libutils)
