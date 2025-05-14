#include <xrtailor/runtime/scene/Scene.hpp>

#include <xrtailor/runtime/scene/WatchDog.hpp>

namespace XRTailor {

void Scene::ClearCallbacks() {
  on_enter.Clear();
  on_exit.Clear();
}

void Scene::SpawnCameraAndLight(GameInstance* game) {
  // camera
  auto camera = SpawnCamera(game);
  camera->Initialize(glm::vec3(0.35, 3.3, 7.2), glm::vec3(1), glm::vec3(-21, 2.25, 0));

  // light
  auto light = SpawnLight(game);
  light->Initialize(glm::vec3(2.5f, 5.0f, 2.5f), glm::vec3(0.2f), glm::vec3(20, 30, 0));
  auto light_comp = light->GetComponent<Light>();
  LOG_DEBUG("Spawn light done");
  SpawnDebug(game);
}

void Scene::SpawnDebug(GameInstance* game) {
  auto quad = game->CreateActor("Debug Quad");
  {
    auto debug_mat = Resource::LoadMaterial("_ShadowDebug");
    {
      float near_plane = 1.0f, far_plane = 7.5f;
      debug_mat->SetFloat("near_plane", near_plane);
      debug_mat->SetFloat("far_plane", far_plane);
      debug_mat->SetTexture("depthMap", game->DepthFrameBuffer());
    }
    std::vector<Scalar> quad_vertices = {
        // positions        // texture Coords
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f,
        0.0f,  0.0f,  1.0f, 1.0f, 0.0f, 1.0f,  1.0f,

        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  -1.0f, 0.0f,
        1.0f,  0.0f,  1.0f, 1.0f, 0.0f, 1.0f,  1.0f,
    };
    std::vector<unsigned int> attributes = {3, 2};
    auto quad_mesh = std::make_shared<Mesh>(attributes, quad_vertices);
    std::shared_ptr<MeshRenderer> renderer(new MeshRenderer(quad_mesh, debug_mat));
    quad->AddComponent(renderer);
    renderer->enabled = false;

    game->on_god_update.Register([renderer]() {
      if (Global::input->GetKeyDown(GLFW_KEY_X)) {
        LOG_INFO("Visualize shadow texutre. Turn on/off by key X");
        renderer->enabled = !renderer->enabled;
      }
    });
  }
}

std::shared_ptr<Actor> Scene::SpawnDebugger(GameInstance* game) {
  auto debugger = game->CreateActor("Debugger");

  return debugger;
}

std::shared_ptr<Actor> Scene::SpawnWatchDogActor(GameInstance* game) {
  auto actor = game->CreateActor("WatchDog");

  std::shared_ptr<WatchDog> watch_dog = std::make_shared<WatchDog>();
  actor->AddComponents({watch_dog});

  return actor;
}

std::shared_ptr<Actor> Scene::SpawnSphere(GameInstance* game, const std::string& name,
                                          glm::vec3 color) {
  auto sphere = game->CreateActor(name);
  MaterialProperty material_property;
  material_property.pre_rendering = [color](Material* mat) {
    mat->SetVec3("material.tint", color);
    mat->SetBool("material.useTexture", false);
  };

  auto material = Resource::LoadMaterial("_Default");

  filesystem::path path = GetHelperDirectory();
  path.append("sphere.obj");
  auto mesh = Resource::LoadMeshDataAndBuildStructure(path.string());
  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);
  sphere->AddComponents({renderer});

  return sphere;
}

std::shared_ptr<Actor> Scene::SpawnLight(GameInstance* game) {
  auto actor = game->CreateActor("Prefab Light");

  filesystem::path path = GetHelperDirectory();
  path.append("cylinder.obj");
  auto mesh = Resource::LoadMeshDataAndBuildStructure(path.string());
  auto material = Resource::LoadMaterial("UnlitWhite");
  auto renderer = std::make_shared<MeshRenderer>(mesh, material);
  auto light = std::make_shared<Light>();

  actor->AddComponents({renderer, light});

  return actor;
}

std::shared_ptr<Actor> Scene::SpawnCamera(GameInstance* game) {
  auto actor = game->CreateActor("Prefab Camera");
  auto camera = std::make_shared<Camera>();
  auto controller = std::make_shared<PlayerController>();
  actor->AddComponents({camera, controller});
  return actor;
}

std::shared_ptr<Actor> Scene::SpawnInfinitePlane(GameInstance* game) {
  auto inf_plane = game->CreateActor("Infinite Plane");
  auto mat = Resource::LoadMaterial("InfinitePlane");
  mat->no_wireframe = true;
  // Plane: ax + by + cz + d = 0
  mat->SetVec4("_Plane", glm::vec4(0, 1, 0, 0));

  const std::vector<Vector3> vertices = {Vector3(1, 1, 0), Vector3(-1, -1, 0), Vector3(-1, 1, 0),
                                         Vector3(1, -1, 0)};
  const std::vector<unsigned int> indices = {2, 1, 0, 3, 0, 1};

  auto mesh =
      std::make_shared<Mesh>(vertices, std::vector<Vector3>(), std::vector<Vector2>(), indices);
  auto renderer = std::make_shared<MeshRenderer>(mesh, mat);
  auto collider = std::make_shared<Collider>(1);
  //auto collider = std::make_shared<Collider>(ColliderType::Plane);
  inf_plane->AddComponents({renderer, collider});

  return inf_plane;
}

std::shared_ptr<Actor> Scene::SpawnColoredCube(GameInstance* game, glm::vec3 color) {
  auto cube = game->CreateActor("Cube");
  auto material = Resource::LoadMaterial("_Default");

  MaterialProperty material_property;
  material_property.pre_rendering = [color](Material* mat) {
    mat->SetVec3("material.tint", color);
    mat->SetBool("material.useTexture", false);
  };

  filesystem::path path = GetHelperDirectory();
  path.append("cube.obj");
  auto mesh = Resource::LoadMeshDataAndBuildStructure(path.string());
  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);
  auto collider = std::make_shared<Collider>(2);
  cube->AddComponents({renderer, collider});

  return cube;
}

}  // namespace XRTailor