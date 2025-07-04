#pragma once

enum TAILOR_EXIT : int {
  SUCCESS = 0,
  // config 1000-1999
  INVALID_ENGINE_CONFIG = 1000,
  INVALID_SIMULATION_CONFIG = 1001,
  INVALID_CLOTH_CONFIG = 1002,
  // CLI 2000-2999
  MISSING_ARGUMENT = 2000,
  ARGUMENT_PARSING_ERROR = 2001,
  // sequence 3000-3999
  INVALID_TARGET_FRAMERATE = 3000,
  SMPLX_ASSERTION_FAILED = 3001,
  SEQUENCE_NOT_FOUND = 3002,
  // asset 4000-4999
  MASK_READ_ERROR = 4000,
  // shader 5000-5999
  SHADER_READ_ERROR = 5000,
  SHADER_NOT_FOUND = 5001,
  SHADER_COMPILATION_ERROR = 5002,
  SHADER_LINKING_ERROR = 5003,
  // logger 6000-6999
  LOGGER_INITIALIZATION_FAILED = 6000,
  // filesystem 7000-7999
  CREATE_DIRECTORY_ERROR = 7000,
  // cuda 8000-8999
  CUDA_INTERNAL_ERROR = 8000,
  NODE_POOL_RUN_OUT = 8001,
  VERTEX_POOL_RUN_OUT = 8002,
  EDGE_POOL_RUN_OUT = 8003,
  FACE_POOL_RUN_OUT = 8004,
  // IO
  FRAME_OUT_OF_RANGE = 9000,
};
