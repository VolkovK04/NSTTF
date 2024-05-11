file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(tensor ${SOURCES})

target_include_directories(tensor PUBLIC SYSTEM ../../libs/)

target_link_libraries(tensor PUBLIC libgpu)