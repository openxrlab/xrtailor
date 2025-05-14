cmake_minimum_required(VERSION 3.16)

project(XRTailor VERSION 1.8.0 LANGUAGES CXX C)

option(XRTAILOR_USE_DOUBLE "Whether use double-precision floating-point arithmetic" OFF)
option(XRTAILOR_BUILD_TESTS "Build tests" OFF)
option(XRTAILOR_BUILD_DOCS "Build documentation" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()


if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()
if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif()
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    find_package(CUDA REQUIRED)  # 查找 CUDA 工具链
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(CMAKE_CUDA_STANDARD 17)
    else()
        set(CMAKE_CUDA_STANDARD 14)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    
    message(STATUS "CUDA_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    #
    # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    # 
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    list(APPEND CUDA_NVCC_FLAGS -v)
    
    string(APPEND CMAKE_CUDA_FLAGS
        " --expt-extended-lambda"
        " -Xcudafe"
        " \"--display_error_number --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=3060 --diag_suppress=20012\""
        " -lineinfo"
    )
else()
    message(FATAL_ERROR "CUDA compiler not found. Please install CUDA toolkit.")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(XRTAILOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/xrtailor/src")
set(XRTAILOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/xrtailor/include")
set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install")

file(GLOB_RECURSE GLAD_FILES 
    "${VENDOR_DIR}/glad/src/*.c"
    "${VENDOR_DIR}/glad/src/*.cpp"
)

set(IMGUI_DIR "${VENDOR_DIR}/imgui")
file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/*.cpp)

file(GLOB_RECURSE PROJECT_HEADERS
    "${XRTAILOR_INCLUDE_DIR}/*.cuh"
    "${XRTAILOR_INCLUDE_DIR}/*.h"
    "${XRTAILOR_INCLUDE_DIR}/*.hpp"
)

file(GLOB_RECURSE PROJECT_SOURCES
    "${XRTAILOR_SOURCE_DIR}/*.cpp"
    "${XRTAILOR_SOURCE_DIR}/*.cu"
    "${XRTAILOR_SOURCE_DIR}/*.vert"
    "${XRTAILOR_SOURCE_DIR}/*.frag"
    "${XRTAILOR_SOURCE_DIR}/*.geom"
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_SOURCES}
        ${PROJECT_HEADERS}
        ${IMGUI_SOURCES}
        ${GLAD_FILES}
)

target_precompile_headers(${PROJECT_NAME}
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${XRTAILOR_INCLUDE_DIR}/xrtailor/core/Precompiled.h>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${XRTAILOR_INCLUDE_DIR}/xrtailor/core/Precompiled.h>"
)

configure_file(
    "${XRTAILOR_INCLUDE_DIR}/xrtailor/core/Version.hpp.in"
    "${XRTAILOR_INCLUDE_DIR}/xrtailor/core/Version.hpp"
    @ONLY
)

set(DEPENDENCY_INCLUDE_DIRS
    "${INSTALL_DIR}/alembic/include"
    "${INSTALL_DIR}/cnpy/include"
    "${INSTALL_DIR}/glfw/include"
    "${INSTALL_DIR}/imath/include/Imath"
    "${INSTALL_DIR}/jsoncpp/include"
    "${INSTALL_DIR}/spdlog/include"
    "${INSTALL_DIR}/zlib/include"
    "${VENDOR_DIR}/glm"
    "${VENDOR_DIR}/glad/include"
    "${VENDOR_DIR}/cxxopts/include"
    "${VENDOR_DIR}/eigen"
    "${VENDOR_DIR}/vcglib"
    "${VENDOR_DIR}/tinygltf"
    "${VENDOR_DIR}/stb_image"
    "${VENDOR_DIR}/cuda"
    "${IMGUI_DIR}"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${XRTAILOR_INCLUDE_DIR}>
        ${CUDA_INCLUDE_DIRS}
    PRIVATE
        ${DEPENDENCY_INCLUDE_DIRS}
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(PLATFORM_LIBS
        "${INSTALL_DIR}/alembic/lib/libAlembic.so"
        "${INSTALL_DIR}/cnpy/lib/libcnpy.a"
        "${INSTALL_DIR}/glfw/lib/libglfw3.a"
        "${INSTALL_DIR}/imath/lib/libImath-3_2.so"
        "${INSTALL_DIR}/jsoncpp/lib/libjsoncpp.a"
        "${INSTALL_DIR}/spdlog/lib/libspdlog.a"
        "${INSTALL_DIR}/zlib/lib/libz.a"
        pthread
        stdc++fs
    )
elseif(MSVC)
    set(PLATFORM_LIBS
        "${INSTALL_DIR}/alembic/lib/Alembic.lib"
        "${INSTALL_DIR}/cnpy/lib/cnpy.lib"
        "${INSTALL_DIR}/glfw/lib/glfw3.lib"
        "${INSTALL_DIR}/Imath/lib/Imath-3_2.lib"
        "${INSTALL_DIR}/jsoncpp/lib/jsoncpp.lib"
        "${INSTALL_DIR}/spdlog/lib/spdlog.lib"
        "${INSTALL_DIR}/zlib/lib/zlib.lib"
    )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${PLATFORM_LIBS} ${CUDA_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-format-security>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>
        # $<$<COMPILE_LANGUAGE:CXX>:-ftime-report>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-funroll-loops>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g>

    )
elseif(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES 
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VS_DPI_AWARE "On"
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/Oi>
        $<$<COMPILE_LANGUAGE:CXX>:/options:strict>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/Gy>
        $<$<COMPILE_LANGUAGE:CXX>:/MD>
    )
    
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING
        _CRT_SECURE_NO_WARNINGS
    )
endif()

if(CMAKE_CUDA_COMPILER)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

macro(GroupSources curdir)
    file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${curdir} ${CMAKE_CURRENT_SOURCE_DIR}/${curdir}/*)

    foreach(child ${children})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${curdir}/${child})
            GroupSources(${curdir}/${child})
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${CMAKE_CURRENT_SOURCE_DIR}/${curdir}/${child})
        endif()
    endforeach()
endmacro()


GroupSources("xrtailor/include")
GroupSources("xrtailor/src")
