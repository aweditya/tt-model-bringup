// TtClient: manages Tenstorrent device lifecycle and metadata.
//
// Design: Single-device only (device 0). The client owns one device and
// one memory space. We use device 0 exclusively per CLAUDE.md constraints.
//
// The PJRT C API header forward-declares PJRT_Client, PJRT_Device, etc.
// as opaque types. We define their concrete layouts HERE. JAX only ever
// holds pointers to these — it never needs to know the layout.

#ifndef TT_PJRT_CLIENT_H_
#define TT_PJRT_CLIENT_H_

#include "pjrt_c_api.h"
#include <string>
#include <vector>

// ============================================================
// Concrete struct definitions for PJRT opaque types
// ============================================================

// PJRT_Error: wraps an error message and code.
struct PJRT_Error {
  std::string message;
  PJRT_Error_Code code;
};

// PJRT_Event: synchronous events (always immediately ready in Phase 1).
struct PJRT_Event {
  bool ready = true;
  PJRT_Error* error = nullptr;
};

// PJRT_DeviceDescription: metadata about a device.
struct PJRT_DeviceDescription {
  int id;
  std::string kind;        // "Blackhole"
  std::string debug_str;   // "TT Blackhole (device 0)"
  std::string to_string;
};

// PJRT_Memory: a memory space on a device.
struct PJRT_Memory {
  int id;
  std::string kind;        // "device"
  std::string debug_str;
  std::string to_string;
  int kind_id;

  // Back-pointer to owning device
  PJRT_Device* device_ptr = nullptr;
};

// PJRT_Device: a single TT device.
struct PJRT_Device {
  PJRT_DeviceDescription description;
  int local_hardware_id;
  bool is_addressable;

  // Memory space
  PJRT_Memory dram_memory;
  std::vector<PJRT_Memory*> memory_ptrs;
  PJRT_Memory* default_memory_ptr = nullptr;

  // Back-pointer to client
  PJRT_Client* client = nullptr;
};

// PJRT_Client: owns the ttnn device and all associated objects.
struct PJRT_Client {
  // ttnn device handle. Stored as void* to avoid ttnn header dependency.
  // Cast to ttnn::Device* in client.cc.
  void* tt_device = nullptr;

  // Platform metadata
  std::string platform_name;     // "tt"
  std::string platform_version;  // ttnn version string

  // Owned device and memory objects
  PJRT_Device device;

  // Stable pointer arrays for device/memory enumeration
  std::vector<PJRT_Device*> device_ptrs;
  std::vector<PJRT_Memory*> memory_ptrs;
};

// PJRT_Buffer: wraps a ttnn tensor (Phase 2+).
struct PJRT_Buffer {
  void* tensor = nullptr;  // ttnn::Tensor* stored as void*
  PJRT_Buffer_Type element_type = PJRT_Buffer_Type_F32;
  std::vector<int64_t> dims;
  size_t size_bytes = 0;
  PJRT_Device* device = nullptr;
  PJRT_Memory* memory = nullptr;
  bool deleted = false;
};

// PJRT_Executable: metadata about a compiled program.
struct PJRT_Executable {
  std::string name;
  size_t num_outputs = 1;
};

// PJRT_LoadedExecutable: an executable bound to devices.
struct PJRT_LoadedExecutable {
  PJRT_Executable executable;
  PJRT_Client* client = nullptr;
  std::vector<PJRT_Device*> addressable_device_ptrs;
  bool deleted = false;
};

// ============================================================
// Client lifecycle functions (called from plugin.cc)
// ============================================================

PJRT_Error* TtClientCreate(PJRT_Client_Create_Args* args);
PJRT_Error* TtClientDestroy(PJRT_Client_Destroy_Args* args);

// ============================================================
// Helpers
// ============================================================

PJRT_Error* MakeError(PJRT_Error_Code code, const std::string& message);
PJRT_Event* MakeReadyEvent();

// Returns the size in bytes of a single element of the given type.
// Returns 0 for unsupported types.
size_t PjrtBufferTypeSize(PJRT_Buffer_Type type);

#endif  // TT_PJRT_CLIENT_H_
