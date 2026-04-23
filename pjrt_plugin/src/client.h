// TtClient: manages Tenstorrent device lifecycle and metadata.
//
// Design: Single-device only (device 0). The client owns one TtDevice and
// one TtMemory, and exposes them through PJRT's device discovery API.
// We use device 0 exclusively per CLAUDE.md constraints.

#ifndef TT_PJRT_CLIENT_H_
#define TT_PJRT_CLIENT_H_

#include "pjrt_c_api.h"
#include <string>
#include <vector>

// Forward declare ttnn types to avoid pulling the full header into client.h.
// The actual ttnn device pointer is stored as void* and cast in client.cc.
// This keeps compilation fast and avoids ttnn header dependency in the header.

// Internal memory space descriptor
struct TtMemory {
  int id;
  std::string kind;        // "device" for DRAM, "l1" for L1 SRAM
  std::string debug_str;
  std::string to_string;

  // Back-pointer to owning device (as PJRT_Device*)
  PJRT_Device* device_ptr;
};

// Internal device descriptor
struct TtDeviceDescription {
  int id;
  std::string kind;        // "Blackhole"
  std::string debug_str;   // "TT Blackhole P150 (device 0)"
  std::string to_string;   // same as debug_str
};

// PJRT_Device is an opaque type. We define its concrete layout here.
// JAX holds PJRT_Device* pointers and calls device-related PJRT functions
// which receive the pointer and extract fields.
struct PJRT_Device {
  TtDeviceDescription description;
  int local_hardware_id;
  bool is_addressable;

  // Memory spaces associated with this device
  TtMemory dram_memory;
  std::vector<PJRT_Memory*> memory_ptrs;  // pointers into dram_memory
  PJRT_Memory* default_memory_ptr;
};

struct PJRT_Memory {
  TtMemory* inner;  // points to TtMemory owned by PJRT_Device
};

struct PJRT_DeviceDescription {
  TtDeviceDescription* inner;  // points to TtDeviceDescription in PJRT_Device
};

// PJRT_Client owns the ttnn device and all associated objects.
struct PJRT_Client {
  // ttnn device handle. Stored as void* to avoid ttnn header dependency here.
  // Cast to ttnn::Device* in client.cc.
  void* tt_device;

  // Platform metadata
  std::string platform_name;    // "tt"
  std::string platform_version; // ttnn version string

  // Owned device and memory objects
  PJRT_Device device;
  PJRT_Memory memory_obj;
  PJRT_DeviceDescription device_desc_obj;

  // Stable pointer arrays for PJRT_Client_Devices / AddressableDevices
  std::vector<PJRT_Device*> device_ptrs;
};

// PJRT_Event: synchronous events (always immediately ready).
// Phase 1 is fully synchronous -- no async execution.
struct PJRT_Event {
  bool ready;
  PJRT_Error* error;  // nullptr if no error
};

// PJRT_Error: wraps an error message and code.
struct PJRT_Error {
  std::string message;
  PJRT_Error_Code code;
};

// ============================================================
// Client lifecycle functions (called from plugin.cc)
// ============================================================

// Create a new client, opening ttnn device 0.
// Returns nullptr on success, PJRT_Error* on failure.
PJRT_Error* TtClientCreate(PJRT_Client_Create_Args* args);

// Destroy the client, closing the ttnn device.
PJRT_Error* TtClientDestroy(PJRT_Client_Destroy_Args* args);

// ============================================================
// Helper: create a PJRT_Error
// ============================================================
PJRT_Error* MakeError(PJRT_Error_Code code, const std::string& message);

// ============================================================
// Helper: create a ready PJRT_Event
// ============================================================
PJRT_Event* MakeReadyEvent();

#endif  // TT_PJRT_CLIENT_H_
