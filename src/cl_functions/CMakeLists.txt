file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(cl_funcs ${SOURCES})