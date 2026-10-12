add_library(xpertstream STATIC
    agent.cpp
    memory.cpp
    interface_smoke_test.cpp
)

target_include_directories(xpertstream PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(xpertstream PUBLIC cxx_std_20)

add_executable(demo_runner ../examples/demo_runner.cpp)
target_link_libraries(demo_runner PRIVATE xpertstream)


