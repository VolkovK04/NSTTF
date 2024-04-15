file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(cl_funcs ${SOURCES})

target_include_directories(cl_funcs PUBLIC ../ ../../libs/)

target_link_libraries(cl_funcs PUBLIC libclew libgpu libutils
    ${PROJECT_NAME}_operation ${PROJECT_NAME}_tensor ${PROJECT_NAME}_utils)

