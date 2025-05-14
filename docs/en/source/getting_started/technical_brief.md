# Technical Brief

- [Cloth Mechanics](md-cloth-mechanics)
- [Constraint Resolve](md-constraint-resolve)
- [Collision Detection](md-collision-detection)
- [Signed Distance Function(SDF) Collision](md-sdf-collision)
- [Repulsion](md-repulsion)
- [Detangle](md-detangle)

This page illustrates the techniques mainly used in *XRTailor*.

(md-cloth-mechanics)=

## Cloth Mechanics

Implicit integration<sup><a href="#ref1">[1]</a></sup> is the most widely used technique in clothing dynamics. However, such force-based method is hard to implement, computational expensive and complex to control. Instead, *XRTailor* employs Position-Based Dynamics (PBD)<sup><a href="#ref2">[2]</a></sup>, a simple, efficient, and robust alternative. Given that cloth stiffness can vary with iteration count, we incorporate Extended Position-Based Dynamics (XPBD)<sup><a href="#ref3">[3]</a></sup> to enhance performance, implementing constraints such as stretch and bending. Since most of real-time applications have limited computational budget, the long range attachments<sup><a href="#ref4">[4]</a></sup> is introduced to achieve better strain limit. To simulate the tether effects, we also implemented the binding constraints.

Additionally, the original XPBD does not support complex accurate material behavior like anisotropy. We introduce finite element methods(FEM)<sup><a href="#ref5">[5]</a></sup> into the simulation framework to address the issue.

(md-constraint-resolve)=

## Constraint Resolve

Solving constraints on GPU is a non-trivial task. A constraint may have shared unknowns. When solving it on CPU, the Gauss-Seidel fashion is a common choice. Regarding the GPU, the Jacobi fashion<sup><a href="#ref6">[6]</a></sup> is simple to implement whereas has convergency and potential overshooting issues. *XRTailor* adopts the Multi-Color Gauss-Seidel<sup><a href="#ref7">[7]</a></sup> fashion to execute constraint resolve on GPU. 

Besides, techniques like small steps<sup><a href="#ref8">[8]</a></sup> and Chebyshev accelerate<sup><a href="#ref9">[9]</a></sup> further speed up the convergency.

(md-collision-detection)=

## Collision Detection

Collision handling consists of broad-phase culling and narrow-phase testing. In the broad phase, data structures that partition the object or world are used to filter out unnecessary elementary tests. *XRTailor* integrates parallel LBVH<sup><a href="#ref10">[10]</a></sup> and spatial hash<sup><a href="#ref11">[11]</a></sup> for efficient filtering. When traversing the BVH, a stack is used to store the mid-step results. However, such stack may overflow when the level of BVH is deep. We use stackless traversal<sup><a href="#ref12">[12]</a></sup> to avoid the problem.

In the narrow phase, a series of primitive tests are performed. For discrete collision detection(DCD), the collision of vertex-face(VF) and edge-edge(EE) pair are checked at a discrete timestamp<sup><a href="#ref13">[13]</a></sup>. For continuous collision detection(CCD), time of impacts(ToI) is evaluated within a time interval. We use Floating-Point Root Finder(FPRF)<sup><a href="#ref14">[14]</a></sup> to estimate ToI. Since FPRF may produce false negatives<sup><a href="#ref15">[15]</a></sup>, *XRTailor* allows adjustments of floating point arithmetic. Based on our observation, the artifacts are negligible when using double-precision floating point arithmetic.

Notice that even if many unnecessary tests are filtered during the broad phase, there also exists duplicated tests at narrow phase. Therefore, the representative triangle<sup><a href="#ref16">[16]</a></sup> technique and orphan test<sup><a href="#ref17">[17]</a></sup> are used to further accelerate the detection.

(md-sdf-collision)=

## SDF Collision

Taking a function as metric, SDF tells whether a point is inside the boundary, which is very suitable for handing cloth collisions with ground or implicit shape models. In *Swift Mode*, we use SDF to handle cloth self collisions. Such particle-based method<sup><a href="#ref7">[7]</a></sup> runs super fast and is very suitable for real-time applications, e.g., video games, crowd simulation. However, this approach has limitations in controlling collision thickness and handling high-speed motion, which can result in missing collisions.

## Mesh-based Collision
  
- ***Swift Mode***: Used to handle obstacle-cloth contact. We follow the PBD collision schema<sup><a href="#ref2">[2]</a></sup> to create signed distance constraints for the VF and EE pairs.

- ***Quality Mode***: Adopts impact zone optimization<sup><a href="#ref18">[18]</a></sup> method to handle collisions. Compared with particle-based collision, the method is able to handle complex collision such as layered cloth with controllable thickness. The technique is used by the well-known cloth simulator ARCSim<sup><a href="#ref19">[19]</a></sup> whereas it's a CPU implementation. Following the method described in I-Cloth<sup><a href="#ref20">[20]</a></sup>, we formulate the collision as a non-linear optimization problem and solved it using backtracking line search<sup><a href="#ref21">[21]</a></sup> method. Besides, we accelerated the performance using the GPU shared memory.

Additionally, *XRTailor* supports Predictive Contact<sup><a href="#ref22">[22]</a></sup> algorithm, commonly used in game engines, to preemptively handle collisions.

(md-repulsion)=

## Repulsion

Resolving thousands of collisions that can readily occur can be prohibitively expensive. We implemented two schemas to detect and resolve potential collisions to reduce the size of impact zones:

  - ***Imminent Repulsion***: A DCD method that tries to prevent collisions from happening in the near future. Once two proximities are close enough, a positional update is applied to the cloth vertices to prevent the penetration. 
  
  - ***Realized Repulsion***: Checks the states between current position and advanced position. Hard constraints will be added if there exists contact.

Based on our observations, the repulsion step dramatically reduces the number of collisions, making the impact zone optimization tractable and efficient.

(md-detangle)=

## Detangle

Collision handling method assume that clothes have no self interpenetration at initial state. Unfortunately, some cases may not have an ideal initial configuration. Untangling cloth<sup><a href="#ref23">[23]</a></sup> is a DCD technique that iteratively push the cloth vertices outside the boundary to resolve the tangling state. However, it cannot handle open contours and it is a non-trivial task to implement it on GPU. Instead, we implemented untangling technique as described in <a href="#ref24">[24]</a> with refined gradient evaluation<sup><a href="#ref25">[25]</a></sup>. Since local scheme converges slowly, we employ global intersection analysis<sup><a href="#ref23">[23]</a></sup> to make the algorithm more efficient.

## References

[1]: <span name = "ref1">Baraff D, Witkin A. Large steps in cloth simulation[M]//Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 2023: 767-778.</span>

[2]: <span name = "ref2">Müller M, Heidelberger B, Hennix M, et al. Position based dynamics[J]. Journal of Visual Communication and Image Representation, 2007, 18(2): 109-118.</span>

[3]: <span name = "ref3">Macklin M, Müller M, Chentanez N. XPBD: position-based simulation of compliant constrained dynamics[C]//Proceedings of the 9th International Conference on Motion in Games. 2016: 49-54.</span>

[4]: <span name = "ref4">Kim T Y, Chentanez N, Müller-Fischer M. Long range attachments-a method to simulate inextensible clothing in computer games[C]//Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2012: 305-310.</span>

[5]: <span name = "ref5">Bender J, Koschier D, Charrier P, et al. Position-based simulation of continuous materials[J]. Computers & Graphics, 2014, 44: 1-10.</span>

[6]: <span name = "ref6">Macklin M, Müller M, Chentanez N, et al. Unified particle physics for real-time applications[J]. ACM Transactions on Graphics (TOG), 2014, 33(4): 1-12.</span>

[7]: <span name = "ref7">Fratarcangeli M, Tibaldo V, Pellacini F. Vivace: a practical gauss-seidel method for stable soft body dynamics[J]. ACM Trans. Graph., 2016, 35(6): 214:1-214:9.</span>

[8]: <span name = "ref8">Macklin M, Storey K, Lu M, et al. Small steps in physics simulation[C]//Proceedings of the 18th Annual ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2019: 1-7.</span>

[9]: <span name = "ref9">Wang H. A chebyshev semi-iterative approach for accelerating projective and position-based dynamics[J]. ACM Transactions on Graphics (TOG), 2015, 34(6): 1-9.</span>

[10]: <span name = "ref10">Karras, Tero. "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees." Proceedings of the Fourth ACM SIGGRAPH/Eurographics conference on High-Performance Graphics. 2012.</span>

[11]: <span name = "ref11">Tang M, Liu Z, Tong R, et al. PSCC: Parallel self-collision culling with spatial hashing on GPUs[J]. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 2018, 1(1): 1-18.</span>

[12]: <span name = "ref12">Damkjær J. Stackless BVH collision detection for physical simulation[J]. University of Copenhagen Universitetsparken: København, Denmark, 2007.</span>

[13]: <span name = "ref13">Ericson C. Real-time collision detection[M]. Crc Press, 2004.</span>

[14]: <span name = "ref14">Provot X. Collision and self-collision handling in cloth model dedicated to design garments[C]//Computer Animation and Simulation’97: Proceedings of the Eurographics Workshop in Budapest, Hungary, September 2–3, 1997. Vienna: Springer Vienna, 1997: 177-189.</span>

[15]: <span name = "ref15">Wang, Bolun, et al. "A large-scale benchmark and an inclusion-based algorithm for continuous collision detection." ACM Transactions on Graphics (TOG) 40.5 (2021): 1-16.</span>

[16]: <span name = "ref16">Curtis S, Tamstorf R, Manocha D. Fast collision detection for deformable models using representative-triangles[C]//Proceedings of the 2008 symposium on Interactive 3D graphics and games. 2008: 61-69.</span>

[17]: <span name = "ref17">Tang M, Curtis S, Yoon S E, et al. Interactive continuous collision detection between deformable models using connectivity-based culling[C]//Proceedings of the 2008 ACM symposium on Solid and physical modeling. 2008: 25-36.</span>

[18]: <span name = "ref18">Bridson R, Fedkiw R, Anderson J. Robust treatment of collisions, contact and friction for cloth animation[C]//Proceedings of the 29th annual conference on Computer graphics and interactive techniques. 2002: 594-603.</span>

[19]: <span name = "ref19">Narain R, Samii A, O'brien J F. Adaptive anisotropic remeshing for cloth simulation[J]. ACM transactions on graphics (TOG), 2012, 31(6): 1-10.</span>

[20]: <span name = "ref20">Tang M, Wang T, Liu Z, et al. I-Cloth: Incremental collision handling for GPU-based interactive cloth simulation[J]. ACM Transactions on Graphics (TOG), 2018, 37(6): 1-10.</span>

[21]: <span name = "ref21">Nocedal J, Wright S. Numerical Optimization[M]. Springer Science & Business Media, 2006.</span>

[22]: <span name = "ref22">Lewin C. Cloth Self Collision with Predictive Contacts[C]//Game Developers Conference. 2018.</span>

[23]: <span name = "ref23">Baraff D, Witkin A, Kass M. Untangling cloth[J]. ACM Transactions on Graphics (TOG), 2003, 22(3): 862-870.</span>

[24]: <span name = "ref24">Volino P, Magnenat-Thalmann N. Resolving surface collisions through intersection contour minimization[J]. ACM Transactions on Graphics (TOG), 2006, 25(3): 1154-1159.</span>

[25]: <span name = "ref25">Ye J, Zhao J. The intersection contour minimization method for untangling oriented deformable surfaces[C]//Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2012: 311-316.</span>
