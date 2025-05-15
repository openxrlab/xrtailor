# Build From Source

- [Download Dependencies](md-download-dependencies)
- [Windows](md-windows)
- [Linux](md-linux)

Building *XRTailor* from source is a non-trivial task. We recommend using our binary distribution, which can be downloaded from release page.

(md-download-dependencies)=

## Download Dependencies

The following third party libraries are required:

```text
alembic: for data exchange
cnpy: for loading .npz files
cxxopts: for command line option parsing
eigen: for linear algebrea
glad: OpenGL interface
glfw: for OpenGL context, mouse/keyboard event, etc.
glm: for 3D mathematics
imath: a math library, which is a dependency for alembic
imgui: for graphical user interface
jsoncpp: for json formating
spdlog: for logging
tinygltf: for gltf loading
vcglib: for triangle meshes manipulation
zlib: for data compression
```


Firstly, clone the repo:

```shell
git clone https://github.com/openxrlab/xrtailor.git

cd xrtailor
```

Third-party libraries are managed as Git submodules. To download all dependencies with appropriate versions, run:

```shell
git submodule update --init --recursive
```

This will automatically fetch and initialize all required submodules.

(md-windows)=

## Windows

### System Requirements

We built our system using the configuration below:

```
Operation System: Windows 10
Visual Studio version: Visual Studio 2019
Windows SDK version: 10.0.19041.0
Nvidia driver version: 528.49
CUDA version: 11.3
cmake version: 3.26.0-rc2
```

### Install Dependencies

First, make sure that you have cmake installed. Then, change the working directory to ```${PROJECT_ROOT}```. Install 3rd party libraries utilizing the following script:

```shell
./scripts/build_dependencies.bat
```

This will automatically compile and install 3rd party libraries used by XRTailor. The dependencies will be installed under ```${PROJECT_ROOT}\install\```.

### Build

Once you have all dependencies successfully installed, change the current working directory to project root and build the project:

```shell
cmake -S . -B build -G "Visual Studio 16 2019"
cmake --build build --config Release
```

If everything is OK, you should see ```XRTailor.exe``` on ```${PROJECT_ROOT}\build\bin\Release```.

(md-linux)=

## Linux

### System Requirements

We built our system using the configuration below:

```
Operation System: Ubuntu 20.04
gcc version: 8.5
Nvidia driver version: 528.49
CUDA version: 11.7
cmake version: 3.26.0-rc2
```

### (Optional)Install Ninja

We employ Ninja as the build tools. To install Ninja on Ubuntu:

```shell
sudo apt update
sudo apt install ninja-build
```

### (Optional)Update cmake

The CMakeLists.txt has a minimum cmake version requirement(>=3.16). You need to upgrade cmake first if your cmake version is out of date. Here are some instructions:

```shell
# download cmake
wget https://cmake.org/files/v3.26/cmake-3.26.0-rc2.tar.gz

# unpack
tar -xvzf cmake-3.26.0-rc2.tar.gz

# configure
chmod 777 ./configure
./configure

# make
make

# install
make install

# switch cmake version by soft link
sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1
```

If everything goes well, you should see the following information using ```cmake --version```:

```
# cmake version 3.26.0-rc2
# CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

(Optional) If the following error occurs:

```shell
CMake Error: Could not find CMAKE_ROOT !!!
CMake has most likely not been installed correctly.
Modules directory not found in
/usr/local/bin
Segmentation fault
```

Solve the issue using:

```shell
hash -r
```

### Install Dependencies

Change the working directory to ```${PROJECT_ROOT}```. Then, install 3rd party libraries utilizing the following script:

```shell
sudo bash ./scripts/build_dependencies.sh
```

This will automatically compile and install 3rd party libraries used by XRTailor. The dependencies will be installed under ```${PROJECT_ROOT}/install/```.

### Build

```shell
cmake -S . -B build -G "Ninja"
cmake --build build --config Release
```

You should see ```XRTailor``` on ```${PROJECT_ROOT}/build/bin```.
