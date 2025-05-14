@echo off
setlocal enabledelayedexpansion

:: Set project root directory
set PROJECT_ROOT=%~dp0..
set INSTALL_DIR=%PROJECT_ROOT%\install
set THIRD_PARTY_DIR=%PROJECT_ROOT%\3rd_party

:: Create install directory if not exists
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Define dependencies and their build configurations
set DEPS[0]=glfw
set DEPS[1]=imath
set DEPS[2]=alembic;-DImath_DIR=../../install/imath/lib/cmake/Imath
set DEPS[3]=jsoncpp
set DEPS[4]=zlib
set DEPS[5]=cnpy;-DZLIB_LIBRARY=../../../install/zlib/lib/zlib -DZLIB_INCLUDE_DIR=../../install/zlib/include
set DEPS[6]=spdlog;-DCMAKE_CXX_STANDARD=17 -DSPDLOG_BUILD_EXAMPLE=OFF

goto main
:: Function to build dependency
:build_dependency
set DEP_NAME=%~1
set EXTRA_ARGS=%~2
echo.
echo Building %DEP_NAME%...
echo --------------------------------------

:: Check if already installed
if exist "%INSTALL_DIR%\%DEP_NAME%" (
    dir /b "%INSTALL_DIR%\%DEP_NAME%" >nul 2>&1
    if not errorlevel 1 (
        echo %DEP_NAME% is already installed, skipping...
        exit /b 0
    )
)

if not exist "%THIRD_PARTY_DIR%\%DEP_NAME%" (
    echo Error: Directory %DEP_NAME% not found!
    exit /b 1
)

cd "%THIRD_PARTY_DIR%\%DEP_NAME%"

if exist "build" rd /s /q "build"
mkdir build

cmake -S . -B build -DCMAKE_INSTALL_PREFIX="../../install/%DEP_NAME%" -G "Visual Studio 16 2019" !EXTRA_ARGS!
if %ERRORLEVEL% neq 0 (
    echo CMake configure failed for %DEP_NAME%
    exit /b 1
)

cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo Build failed for %DEP_NAME%
    exit /b 1
)

cmake --install build
if %ERRORLEVEL% neq 0 (
    echo Install failed for %DEP_NAME%
    exit /b 1
)

echo %DEP_NAME% built successfully
echo.
exit /b 0

:: Main execution
:main
echo Starting dependency build process...
echo =====================================

for /L %%i in (0,1,6) do (
    set "entry=!DEPS[%%i]!"
    for /f "tokens=1,* delims=;" %%a in ("!entry!") do (
        set "DEP_NAME=%%a"
        set "EXTRA_ARGS=%%b"
        call :build_dependency !DEP_NAME! "!EXTRA_ARGS!"
        if !ERRORLEVEL! neq 0 (
            echo Build process failed at %%a
            exit /b 1
        )
    )
)

echo.
echo All dependencies built successfully!
echo =====================================

cd "%PROJECT_ROOT%"