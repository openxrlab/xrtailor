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
*XRTailor* delivers high-fidelity cloth dynamics while maintaining performance, making it a practical choice for applications in animation, gaming and machine learning dataset synthesis.

## Features

- *Realistic Cloth Mechanics*. *XRTailor* models the physical behavior of fabrics, incorporating key mechanical properties such as stretch, bending, and anisotropy to provide plausible cloth deformation.

<div align="center">
  <video src="https://github.com/user-attachments/assets/2fe3a8e1-cc01-4ba4-80c6-c00f4cb839cb" width="50%"> </video>
</div>

- *Collisions*. Collision detection and response are essential for cloth simulation. *XRTailor* supports obstacle-cloth collision, environment-cloth collision and self-collision. These features help maintain natural interactions between cloth and surrounding objects.

- *Fully Parallelized*. To achieve better performance, *XRTailor* employs advanced data structures and algorithms specifically designed for GPU execution. By maximizing parallelism, the engine supports rapid computation, making it suitable for real-time and offline simulations alike.

- *Balanced Performance Modes*. *XRTailor* offers two modes to accommodate different needs:

    - ***Swift Mode***: Optimized for real-time applications, offering rapid simulations with simplified fabric properties and collision handling.

    - ***Quality Mode***: Prioritizes accuracy, delivering highly detailed simulations at the cost of increased computational overhead.

<div align="center">
  <video src="https://github.com/user-attachments/assets/0e3d986e-1bdc-40fb-b916-ce52afdbb930" width="50%"> </video>
</div>

- *Fully Automatic*. Unlike existing cloth simulators, animators do not need to place the cloth pieces in appropriate positions to dress an avatar.

- *Highly Compatible with SMPL(X)*. *XRTailor* supports SMPL, SMPLH, SMPLX with AMASS integration.

<div align="center">
  <video src="https://github.com/user-attachments/assets/faffb237-c3f3-4f09-bf2f-47ff7c1bec0a" width="50%"> </video>
</div>

- *GLTF Support*. Allows importing mannequins with skeletal animation in GLTF format.

<div align="center">
  <video src="https://github.com/user-attachments/assets/015f36c9-1bc6-4344-ac95-d6ad7975276b" width="50%"> </video>
</div>

- *Easy to Use*. Traditional cloth simulation workflow is laborious and knowledge intensive. *XRTailor* aims to simplify the process, allowing users to obtain desired outputs (such as Alembic or OBJ sequences) using a single command.

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
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

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
- [XRViewer](https://github.com/openxrlab/xrviewer): OpenXRLab Data Visualization Toolbox.

## References

[1]: <span name = "ref_lsc">Baraff D, Witkin A. Large steps in cloth simulation[M]//Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 2023: 767-778.</span>

[2]: <span name = "ref_pbd">Müller M, Heidelberger B, Hennix M, et al. Position based dynamics[J]. Journal of Visual Communication and Image Representation, 2007, 18(2): 109-118.</span>

[3]: <span name = "ref_xpbd">Macklin M, Müller M, Chentanez N. XPBD: position-based simulation of compliant constrained dynamics[C]//Proceedings of the 9th International Conference on Motion in Games. 2016: 49-54.</span>

[4]: <span name = "ref_lra">Kim T Y, Chentanez N, Müller-Fischer M. Long range attachments-a method to simulate inextensible clothing in computer games[C]//Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2012: 305-310.</span>

[5]: <span name = "ref_psc">Bender J, Koschier D, Charrier P, et al. Position-based simulation of continuous materials[J]. Computers & Graphics, 2014, 44: 1-10.</span>

[6]: <span name = "ref_upp">Macklin M, Müller M, Chentanez N, et al. Unified particle physics for real-time applications[J]. ACM Transactions on Graphics (TOG), 2014, 33(4): 1-12.</span>

[7]: <span name = "ref_vivace">Fratarcangeli M, Tibaldo V, Pellacini F. Vivace: a practical gauss-seidel method for stable soft body dynamics[J]. ACM Trans. Graph., 2016, 35(6): 214:1-214:9.</span>

[8]: <span name = "ref_ss">Macklin M, Storey K, Lu M, et al. Small steps in physics simulation[C]//Proceedings of the 18th Annual ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2019: 1-7.</span>

[9]: <span name = "ref_chebyshev">Wang H. A chebyshev semi-iterative approach for accelerating projective and position-based dynamics[J]. ACM Transactions on Graphics (TOG), 2015, 34(6): 1-9.</span>

[10]: <span name = "ref_lbvh">Karras, Tero. "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees." Proceedings of the Fourth ACM SIGGRAPH/Eurographics conference on High-Performance Graphics. 2012.</span>

[11]: <span name = "ref_pscc">Tang M, Liu Z, Tong R, et al. PSCC: Parallel self-collision culling with spatial hashing on GPUs[J]. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 2018, 1(1): 1-18.</span>

[12]: <span name = "ref_stackless">Damkjær J. Stackless BVH collision detection for physical simulation[J]. University of Copenhagen Universitetsparken: København, Denmark, 2007.</span>

[13]: <span name = "ref_rcd">Ericson C. Real-time collision detection[M]. Crc Press, 2004.</span>

[14]: <span name = "ref_csh">Provot X. Collision and self-collision handling in cloth model dedicated to design garments[C]//Computer Animation and Simulation’97: Proceedings of the Eurographics Workshop in Budapest, Hungary, September 2–3, 1997. Vienna: Springer Vienna, 1997: 177-189.</span>

[15]: <span name = "ref_lbi">Wang, Bolun, et al. "A large-scale benchmark and an inclusion-based algorithm for continuous collision detection." ACM Transactions on Graphics (TOG) 40.5 (2021): 1-16.</span>

[16]: <span name = "ref_fcd">Curtis S, Tamstorf R, Manocha D. Fast collision detection for deformable models using representative-triangles[C]//Proceedings of the 2008 symposium on Interactive 3D graphics and games. 2008: 61-69.</span>

[17]: <span name = "ref_icc">Tang M, Curtis S, Yoon S E, et al. Interactive continuous collision detection between deformable models using connectivity-based culling[C]//Proceedings of the 2008 ACM symposium on Solid and physical modeling. 2008: 25-36.</span>

[18]: <span name = "ref_bridson">Bridson R, Fedkiw R, Anderson J. Robust treatment of collisions, contact and friction for cloth animation[C]//Proceedings of the 29th annual conference on Computer graphics and interactive techniques. 2002: 594-603.</span>

[19]: <span name = "ref_arcsim">Narain R, Samii A, O'brien J F. Adaptive anisotropic remeshing for cloth simulation[J]. ACM transactions on graphics (TOG), 2012, 31(6): 1-10.</span>

[20]: <span name = "ref_icloth">Tang M, Wang T, Liu Z, et al. I-Cloth: Incremental collision handling for GPU-based interactive cloth simulation[J]. ACM Transactions on Graphics (TOG), 2018, 37(6): 1-10.</span>

[21]: <span name = "ref_no">Nocedal J, Wright S. Numerical Optimization[M]. Springer Science & Business Media, 2006.</span>

[22]: <span name = "ref_pc">Lewin C. Cloth Self Collision with Predictive Contacts[C]//Game Developers Conference. 2018.</span>

[23]: <span name = "ref_untangling">Baraff D, Witkin A, Kass M. Untangling cloth[J]. ACM Transactions on Graphics (TOG), 2003, 22(3): 862-870.</span>

[24]: <span name = "ref_rsc">Volino P, Magnenat-Thalmann N. Resolving surface collisions through intersection contour minimization[J]. ACM Transactions on Graphics (TOG), 2006, 25(3): 1154-1159.</span>

[25]: <span name = "ref_icm">Ye J, Zhao J. The intersection contour minimization method for untangling oriented deformable surfaces[C]//Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2012: 311-316.</span>

[26]: <span name = "ref_descent">Wang H, Yang Y. Descent methods for elastic body simulation on the GPU[J]. ACM Transactions on Graphics (TOG), 2016, 35(6): 1-10.</span>

[27] <span name = "ref_smpl">SMPL: A Skinned Multi-Person Linear Model. Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, Michael J. Black. 2015</span>

[28] <span name = "ref_smplify">Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image. Federica Bogo*, Angjoo Kanazawa*, Christoph Lassner, Peter Gehler, Javier Romero, Michael Black. 2016</span>

[29] <span name = "ref_mano">Embodied Hands: Modeling and Capturing Hands and Bodies Together. Javier Romero*, Dimitrios Tzionas*, and Michael J Black. 2017</span>

[30] <span name = "ref_smplx">Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. G. Pavlakos*, V. Choutas*, N. Ghorbani, T. Bolkart, A. A. A. Osman, D. Tzionas and M. J. Black. 2019</span>

[31] <span name = "ref_amass">AMASS: Archive of Motion Capture as Surface Shapes. Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.</span>