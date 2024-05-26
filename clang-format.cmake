file(GLOB_RECURSE
     ALL_CXX_SOURCE_FILES
     ${PROJECT_SOURCE_DIR}
     src/*.cpp
     src/*.h
     src/*.cl
     tests/*.cpp
     tests/*.h
     )

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND clang-format
    -i
    ${ALL_CXX_SOURCE_FILES}
    )
else()
    message(WARNING "clang-format not found")
endif()