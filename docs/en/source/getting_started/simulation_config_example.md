# Simulation Config Example

- [Format](md-format)
- [Detailed Description](md-detailed-description)

Simulation config is a set of parameters that illustrate the composition of scene and describe how cloth behaves. It is composed of 3 parts:

- pipeline: which pipeline to use. There are 3 pipelines by default: smpl, gltf and universal.

- animation: animation settings, e.g., whether record animations, export format, etc.

- solver: solver settings. There are 2 modes of solver: swift mode and quality mode.

(md-format)=

## Format

A simulation config should follow the structure as shown below:

```json
{
	"PIPELINE_NAME":
	{
		//...
	},
	"ANIMATION":
	{
		//...
	},
	"MODE_NAME":
	{
		//...
	}
}
```

(md-detailed-description)=

## Pipeline

### SMPL

|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- |
|CLOTH\_STYLES|Required|vector<string>|\-|Name of clothes in assets to be simulated.|
|NPZ\_PATH|Required|string|\-|Path to body motion in npz format.|
|BODY\_MODEL|Required|string|\-|The body model used for simulation, e.g., SMPL, SMPLH, SMPLX.|
|NUM\_LERPED\_FRAMES| Optional | int | 0 | Number of frames that transform character from rest-pose to motion.|
|ENABLE\_COLLISION\_FILTER|Optional|bool|false| The collision of hands and head will be ignored when set to True.|

### GLTF

|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- |
|CLOTH\_STYLES|Required|vector<string>|\-|Name of clothes in assets to be simulated.|
|CHARACTER\_NAME|Required|string|\-|Name of the character in assets.|
|GLTF\_PATH|Required|string|-|Path to gltf file that contains keyframe animations.|
|NUM\_LERPED\_FRAMES| Optional | int | 0 | Number of frames that transform character from rest-pose to motion.|

### Universal

|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- |
|CLOTHES|Required|vector<CLOTH\>|\-|Clothes to be added to the scene.|
|OBSTACLES|Required|vector<OBSTACLE\>|\-|Obstacles to be added to the scene.|
|NUM\_FRAMES|Optional|int|100|Number of frames to simulate.|

in which the CLOTH and OBSTACLE are defined as follows:

- CLOTH:

|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- |
|OBJ_PATH|Required|string|\-|Path to the cloth file in obj format.|
|POSITION|Optional|vec3|(0, 0, 0)|Position of the cloth in world space.|
|ROTATION|Optional|vec3|(0, 0, 0)|Euler rotation of the cloth in world space.|
|SCALE|Optional|vec3|(1, 1, 1)|Scale of the cloth in world space.|
|ATTACHED_INDICES|Optional|vector<int\>|[]|Indices of vertices to be fixed during the simulation.|

- OBSTACLE:

|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- |
|OBJ_PATH|Required|string|\-|Path to the obstacle file in obj format.|
|POSITION|Optional|vec3|(0, 0, 0)|Position of the obstacle in world space.|
|ROTATION|Optional|vec3|(0, 0, 0)|Euler rotation of the obstacle in world space.|
|SCALE|Optional|vec3|(1, 1, 1)|Scale of the obstacle in world space.|

## Animation

|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- |
|NUM\_PRE\_SIMULATION\_FRAMES|Optional|int|120|Number of warmup frames to pass-by before simulation.|
|RECORD\_OBSTACLE|Optional|bool|true|Whether record obstacle animation.|
|RECORD\_CLOTH|Optional|bool|true|Whether record cloth animation.|
|EXPORT\_FORMAT|Optional|int|0|Animation file export format. 0: alembic; 1: obj sequence. |
|EXPORT\_DIRECTORY|Required|string|\-|Path to save synthesized animation.|

## Solver Mode

### Swift

|SCOPE|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- | ----- |
| SOLVER | NUM\_SUBSTEPS | Optional | int | 4 |Number of steps that execute within a frame.|
|  |NUM\_ITERATIONS | Optional | int | 10 |Number of solver iterations to perform per-substep.|
|  | MAX\_SPEED | Optional | float | 1e6f | Maximum velocity of particles.|
|  |GRAVITY|Optional|vec3|(0.0f, -9.8f, 0.0f)|Constant acceleration applied to all particles.|
|FABRIC|STRETCH\_COMPLIANCE|Optional|float|0.0f|See Miles Macklin's [blog](http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/). Gravity is strong enough to cause stretching, we recommend using zero compliance on cloth mesh edges with XPBD.|
| |BEND\_COMPLIANCE|Optional|float|10.0f|Larger bend compliance results in larger bend resistance.|
| |RELAXATION\_FACTOR|Optional|float|0.25f|Control the convergence rate of the Jacobi solver. If the value is too small, the solver will not converge, while value that is too large may lead to instability.|
| |LONG\_RANGE\_STRETCHINESS|Optional|float|1.05f|Control the maximum radius that particles can be reached.|
| |GEODESIC\_LRA|Optional|bool|true|Measurement of LRA. true: Geodesic distance; false: Euclidean distance.|
| |SOLVE\_BENDING|Optional|bool|false|Whether solve bending constraint.|
|COLLISION|NUM\_COLLISION\_PASSES|Optional|int|10|Number of interleaved collisions. Higher iteration count results in better convergency with computational overhead.|
||SDF\_COLLISION\_MARGIN|Optional|float|0.01f|Distance particles maintain against shapes.|
| |BVH\_TORLENCE|float|Optional|0.001f|Slightly extend the BVH along the vertex normal.|
| COLLISION / PARTICLE |FRICTION|Optional|float|0.1|Strength of friction.|
||MAX\_NEIGHBOR\_SIZE|Optional|int|64|Maximum number of neighbor particles to cache.|
| |INTER\_LEAVED\_HASH|Optional|int|3|Hash once every n substeps. This can improves performance greatly.|
| |ENABLE\_SELF\_COLLISION|Optional|bool|true|Whether enable self-collision between cloth vertices.|
| |PARTICLE\_DIAMETER|Optional|float|1.5f|Multiply original stretch length by this scalar to obtain particle diameter.|
| |HASH\_CELL\_SIZE|Optional|float|1.5f|Multiply particle diameter by this scalar to obtain hash cell size.|

### Quality

|SCOPE|PARAM|FIELD|TYPE|DEFAULT VALUE|DESCRIPTION|
| :-----: | ----- | ----- | ----- | ----- | ----- |
|SOLVER|NUM\_SUBSTEPS|Optional|int|1|Number of steps that execute within a frame. Note that high step count may lead to instability.|
| |NUM\_ITERATIONS|Optional|int|200|Number of solver iterations to perform per-substep.|
| |MAX\_SPEED|Optional|float|1e6f|Maximum velocity of particles.|
| |GRAVITY|Optional|vec3|(0.0f, -9.8f, 0.0f)|Constant acceleration applied to all particles.|
| |DAMPING|Optional|float|0.98f|Viscous drag force, applies a force proportional, and opposite to the particle velocity.|
|FABRIC|XX\_STIFFNESS|Optional|float|1.0f|Stiffness in xx direction.|
| |XY\_STIFFNESS|Optional|float|1.0f|Stiffness in xy direction.|
| |YY\_STIFFNESS|Optional|float|1.0f|Stiffness in yy direction.|
| |XY\_POISSION\_RATIO|Optional|float|0.3f|Possion ratio in xy direction.|
| |YX\_POISSION\_RATIO|Optional|float|0.3f|Possion ratio in yx direction.|
| |SOLVE\_BENDING|Optional|bool|false|Whether solve bending constraint.|
| |BENDING\_STIFFNESS|Optional|float|1.0f|Stiffness of bending constraint.|
| |LONG\_RANGE\_STRETCHINESS|Optional|float|1.05f|Control the maximum radius that particles can be reached.|
| |GEODESIC\_LRA|Optional|bool|true|Measurement of LRA. true: Geodesic distance; false: Euclidean distance.|
|REPULSION|ENABLE\_IMMINENT\_REPULSION|Optional|bool|true|Whether enable imminent repulsions.|
||IMMINENT_THICKNESS|Optional|float|1e-3f|Thickness of imminent repulsion.|
||ENABLE\_PBD\_REPULSION|Optional|bool|true|Whether enable PBD repulsions.|
||PBD_THICKNESS|Optional|float|1e-3f|Thickness of PBD repulsion.|
| |RELAXATION\_RATE|Optional|float|0.25f|Control the convergence rate of the repulsion solver.|
|IMPACT_ZONE|OBSTACLE\_MASS|Optional|float|1e3f|Mass of obstacle.|
||THICKNESS|Optional|float|1e-4f|Cloth collision thickness.|
