file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(utils ${SOURCES})

target_include_directories(utils PUBLIC ../ ../../libs/)

target_link_libraries(utils PUBLIC
    computationGraph executor tensor operations
    libutils)
