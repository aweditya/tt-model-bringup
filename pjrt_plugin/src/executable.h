// TtExecutable: wraps a parsed StableHLO module for execution.
//
// Phase 1: Stub only. Real implementation in Phase 3 will parse
// StableHLO MLIR and dispatch ops to ttnn.

#ifndef TT_PJRT_EXECUTABLE_H_
#define TT_PJRT_EXECUTABLE_H_

#include "pjrt_c_api.h"
#include <string>
#include <vector>

struct PJRT_Executable {
  std::string name;
  size_t num_outputs;
};

struct PJRT_LoadedExecutable {
  PJRT_Executable executable;
  PJRT_Device* device;
  std::vector<PJRT_Device*> addressable_device_ptrs;
};

#endif  // TT_PJRT_EXECUTABLE_H_
