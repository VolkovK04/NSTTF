file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(${PROJECT_NAME}_utils ${SOURCES})

target_include_directories(${PROJECT_NAME}_utils PUBLIC ../ ../../libs/)

target_link_libraries(${PROJECT_NAME}_utils PUBLIC libclew libgpu libutils
    ${PROJECT_NAME}_operation ${PROJECT_NAME}_tensor ${PROJECT_NAME}_computationGraph cl_funcs)
