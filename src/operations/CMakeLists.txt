file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(operations ${SOURCES})

target_include_directories(operations PUBLIC ../)

target_link_libraries(operations PUBLIC
    tensor compiler libutils)