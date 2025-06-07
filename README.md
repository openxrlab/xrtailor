<br/>

<div align="center">
    <img src="resources/xrtailor-logo.png" width="600"/>
</div>

<br/>

<div align="center">

[![Documentation](https://readthedocs.org/projects/xrtailor/badge/?version=latest)](https://xrtailor.readthedocs.io/en/latest/?badge=latest)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/openxrlab/xrtailor.svg)](https://github.com/openxrlab/xrtailor/issues)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

</div>


## Introduction

*XRTailor* is a GPU-accelerated cloth simulation engine optimized for large-scale data generation. By leveraging parallel computing techniques, 
*XRTailor* delivers high-fidelity cloth dynamics while maintaining performance, making it a practical choice for applications such as animation, gaming and machine learning dataset synthesis.

## Features

- *Realistic Cloth Mechanics*. *XRTailor* models the physical behavior of fabrics, incorporating key mechanical properties such as stretch, bending, and anisotropy to provide plausible cloth deformation.

<div align="center">
  <video src="https://github.com/user-attachments/assets/2fe3a8e1-cc01-4ba4-80c6-c00f4cb839cb" width="50%"> </video>
</div>

- *Collisions*. Collision detection and response are essential for cloth simulation. *XRTailor* supports obstacle-cloth collision, environment-cloth collision and self-collision. These features help maintain natural interactions between cloth and surrounding objects.

- *Fully Parallelized*. To achieve better performance, *XRTailor* employs advanced data structures and algorithms specifically designed for GPU execution. By maximizing parallelism, the engine supports fast computation, making it suitable for real-time and offline simulations alike.

- *Balanced Performance Modes*. *XRTailor* offers two modes to accommodate different needs:

    - ***Swift Mode***: Optimized for real-time applications, offering fast simulations with simplified fabric properties and collision handling.

    - ***Quality Mode***: Prioritizes accuracy, delivering highly detailed simulations at the cost of increased computational overhead.

<div align="center">
  <video src="https://github.com/user-attachments/assets/0e3d986e-1bdc-40fb-b916-ce52afdbb930" width="50%"> </video>
</div>

- *Fully Automatic*. Unlike existing cloth simulators, animators are not required to place cloth pieces in appropriate positions to dress an avatar.

- *Highly Compatible with SMPL(X)*. *XRTailor* supports SMPL, SMPLH, SMPLX with AMASS integration.

<div align="center">
  <video src="https://github.com/user-attachments/assets/faffb237-c3f3-4f09-bf2f-47ff7c1bec0a" width="50%"> </video>
</div>

- *GLTF Support*. *XRTailor* supports importing mannequins with skeletal animation in GLTF format.

<div align="center">
  <video src="https://github.com/user-attachments/assets/015f36c9-1bc6-4344-ac95-d6ad7975276b" width="50%"> </video>
</div>

- *Easy to Use*. Traditional cloth simulation workflows are labor-intensive and require significant expertise. *XRTailor* aims to simplify the process, allowing users to obtain desired outputs (such as Alembic or OBJ sequences) using a single command.

<div align="center">
  <video src="https://github.com/user-attachments/assets/18fc1805-1b8d-4ebf-985d-1a53ac45747a" width="50%"> </video>
</div>

- Simulation as a Service. *XRTailor* is a powerful and scalable platform designed for large-scale data generation. Our simulation service enables users to efficiently create and manage vast amounts of synthetic data. Designed for large-scale synthetic data generation, *XRTailor* can be deployed via Docker, even in headless environments.

<div align="center">
  <video src="https://github.com/user-attachments/assets/051c0946-dcb9-4151-8e60-b03d13a599b9" width="50%"> </video>
</div>

- *Multi Platform Support*. *XRTailor* runs on Windows and Linux systems that support CUDA, offering flexibility across computing environments.

- *OpenGL Rendering*. A built-in graphical interface provides visualization and control over the simulation process.

<div align="center">
  <video src="https://github.com/user-attachments/assets/bb9dd20a-720f-468e-9ee1-2edb0d3937d7" width="50%"> </video>
</div>

## Getting Started

Please refer to our [documentation page](https://xrtailor.readthedocs.io) for more details.

## License

XRTailor is an open source project that is contributed by researchers and engineers from both the academia and the industry.
We appreciate all contributors who implement their methods or add new features, as well as users who give valuable feedback.

The license of our codebase is Apache-2.0, see [LICENSE](LICENSE) for more information. Note that *XRTailor* is developed upon other open-source projects and uses many third-party libraries. Refer to [docs/licenses](docs/licenses/) to view the full licenses list. We would like to pay tribute to open-source implementations to which we rely on.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{xrtailor,
    title={OpenXRLab GPU Cloth Simulator},
    author={XRTailor Contributors},
    howpublished = {\url{https://github.com/openxrlab/xrtailor}},
    year={2025}
}
```

## Contributing

We appreciate all contributions to improve XRTailor. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Projects in OpenXRLab

- [XRPrimer](https://github.com/openxrlab/xrprimer): OpenXRLab foundational library for XR-related algorithms.
- [XRSLAM](https://github.com/openxrlab/xrslam): OpenXRLab Visual-inertial SLAM Toolbox and Benchmark.
- [XRSfM](https://github.com/openxrlab/xrsfm): OpenXRLab Structure-from-Motion Toolbox and Benchmark.
- [XRLocalization](https://github.com/openxrlab/xrlocalization): OpenXRLab Visual Localization Toolbox and Server.
- [XRMoCap](https://github.com/openxrlab/xrmocap): OpenXRLab Multi-view Motion Capture Toolbox and Benchmark.
- [XRMoGen](https://github.com/openxrlab/xrmogen): OpenXRLab Human Motion Generation Toolbox and Benchmark.
- [XRNeRF](https://github.com/openxrlab/xrnerf): OpenXRLab Neural Radiance Field (NeRF) Toolbox and Benchmark.
- [XRFeitoria](https://github.com/openxrlab/xrfeitoria): OpenXRLab Synthetic Data Rendering Toolbox.
- [XRTailor](https://github.com/openxrlab/xrtailor): OpenXRLab GPU Cloth Simulator.
- [XRViewer](https://github.com/openxrlab/xrviewer): OpenXRLab Data Visualization Toolbox.

## References

Our project is inspired by many previous studies, and we are grateful for the development and sharing within the community. Refer to [reference](docs/reference.md) for prior outstanding work.