file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(dataLoader ${SOURCES})

target_include_directories(dataLoader PUBLIC ../)

target_link_libraries(dataLoader PUBLIC tensor)