file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(executor ${SOURCES})

target_include_directories(executor PUBLIC ../)

target_link_libraries(executor PUBLIC utils tensor)