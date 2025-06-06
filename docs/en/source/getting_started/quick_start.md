# Quick Start

- [Download Assets](md-download-assets)
- [Download Binary](md-download-binary)
- [Windows](md-windows)


This tutorial introduces how to run the *XRTailor* program.

(md-download-assets)=

## Download Assets

<!-- ### Structure -->

Assets is an directory that contains necessary files for running *XRTailor*, e.g., garment templates, garment configurations, body models. The assets directory should follow the hierarchy below:

```shell
- Assets
    - Garment
        - SMPLH                                               # SMPL and SMPLH share the same template
            - Template                                        # draped garment meshes in canonical space
                - upper_top_female.obj
                - lower_long-skirt_female.obj
                - ...
            - Config                                          # garment attributes, e.g., attached vertex indices
                - upper_top_female.json
                - lower_long-skirt_female.json
                - ...
        - SMPLX
        - JQR
    - Body
        - Model                                               # body models
            - smpl
                - SMPL_FEMALE.npz                             # model weights
                - ...
            - smplh
                - ...
            - smplx  
                - ...                                     
        - Template                                            
            - SMPLH_female.obj                                # body mesh in canonical space
            - SMPLH_mask.txt                                  # determines whether primitives are used to construct BVH
            - SMPLX_female.obj
            - SMPLX_mask.txt
            - ...
    - ...
```

We provide example assets to test the pipeline, which can be downloaded from [XRTailor Assets](https://drive.google.com/file/d/1uIDpHj3IwgDJpAmNw_X30KlaJTOfRu4r/view?usp=sharing).

### Getting SMPL Models

Note that if you need to use smpl pipeline, you need to manually download the SMPL/SMPL+H/SMPL-X models. See [smpl models setup](https://github.com/sxyu/smplxpp/tree/master/data/models) for more details. The downloaded models should be put under ```Assets/Body/Model```.

(md-download-binary)=

## Download Binary

Download binary distribution from [release page](https://github.com/openxrlab/xrtailor/releases).

(md-windows)=

## Windows

We tested our engine on Windows10 with CUDA 11.3. To execute the binary, make sure that you have CUDA 11.3-12.0 installed.

The binary distribution contains the following files:

```text
- XRTailor_X.X.X_binary_WIN64
    - XRTailor.exe                      // program entry
    - engine_conf.json                  // engine config example
    - Alembic.dll Imath-3_1.dll ...     // runtime libraries for XRTailor, leave as it is
```

To run the simulation, you need to configure both engine config and simulation config. In *engine_conf.json*, replace the following paths:

```shell
{
    "LOG_PATH": ${PATH_TO_YOUR_LOG_DIRECTORY},
    ...
    "ASSET_DIRECTORY": ${PATH_TO_ASSET_DIRECTORY},
    ...
}
```

The simulation config differs when you run different pipelines and solver modes. We provide some simulation config examples, see [smpl pipeline example](https://github.com/openxrlab/xrtailor/tree/main/examples/smpl/), [gltf pipeline example](https://github.com/openxrlab/xrtailor/tree/main/examples/gltf/) and [universal pipeline example](https://github.com/openxrlab/xrtailor/tree/main/examples/universal/) for more details.

There are plenty of tunable parameters in engine config and simulation config, see [engine config example](./engine_config_example.md) and [simulation config example](./simulation_config_example.md) for more details.

Once the engine config and simulation config are properly configured, start the engine utilizing the following command:

```powershell
.\XRTailor.exe --simulation_config=".\simulation_conf.json" --engine_config=".\engine_conf.json"
```

For more example usages:

```powershell
.\XRTailor.exe --help
```

## Linux

We tested our engine on Ubuntu 16.04/18.04 with Nvidia driver 515.65.01, CUDA 11.3. To execute the binary, make sure that you have Nvidia driver 515.65+ and CUDA 11.3-12.0 installed.

The binary distribution contains the following files:

```text
- XRTailor_X.X.X_binary_LINUX64
    - XRTailor                                     // program entry
    - engine_conf.json                           // engine config example
    - libAlembic.so.1.8 librt.so.1 ...           // runtime libraries for XRTailor, leave as it is
```

To run the simulation, you need to configure both engine config and simulation config. In *engine_conf.json*, replace the following paths:

```shell
{
    "LOG_PATH": ${PATH_TO_YOUR_LOG_DIRECTORY},
    ...
    "ASSET_DIRECTORY": ${PATH_TO_ASSET_DIRECTORY},
    ...
}
```

The simulation config differs when you run different pipelines and solver modes. We provide some simulation config examples, see [smpl pipeline example](https://github.com/openxrlab/xrtailor/tree/main/examples/smpl/), [gltf pipeline example](https://github.com/openxrlab/xrtailor/tree/main/examples/gltf/) and [universal pipeline example](https://github.com/openxrlab/xrtailor/tree/main/examples/universal/) for more details.

There are plenty of tunable parameters in engine config and simulation config, see [engine config example](./engine_config_example.md) and [simulation config example](./simulation_config_example.md) for more details.

Once the engine config and simulation config are properly configured, start the engine utilizing the following command:

### Simulation with Display

Once the engine config and simulation config are properly configured, you can execute the simulation utilizing the following command:

```shell
./XRTailor --simulation_config="./simulation_conf.json" --engine_config="./engine_conf.json"
```

For more example usages:

```
./XRTailor --help
```

### Simulation without Display

Sometimes we need to run simulation on a remote server that has no display. Unfortunately, OpenGL need a window to create graphical context. Therefore, you need to setup remote OpenGL following the instructions in [Remote OpenGL Setup without X](https://gist.github.com/shehzan10/8d36c908af216573a1f0) to create a virtual display.

Once you have remote OpenGL properly configured, execute the simulation using:

```bash
env DISPLAY=:0 ./XRTailor --simulation_config="./simulation_conf.json" --engine_config="./engine_conf.json"
```
