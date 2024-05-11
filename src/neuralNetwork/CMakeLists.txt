file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_library(neuralNetwork ${SOURCES})

target_include_directories(neuralNetwork PUBLIC ../)

target_link_libraries(neuralNetwork PUBLIC
    tensor compiler libutils operations executor)