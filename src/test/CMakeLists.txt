include_directories(${PROJECT_SOURCE_DIR}/test)

set(TEST_SOURCES
        #run_tests.h
        #run_tests.cpp
        runtime/local/io/ReadCsvTest.cpp
)

add_executable(run_tests ${TEST_SOURCES})
set_target_properties(run_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)