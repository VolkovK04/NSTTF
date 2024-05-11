file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(computationGraph ${SOURCES})

target_include_directories(computationGraph PUBLIC ../  SYSTEM ../../libs/)

target_link_libraries(computationGraph PUBLIC operations tensor)

