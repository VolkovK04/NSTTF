file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(operations ${SOURCES})