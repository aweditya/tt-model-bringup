// TtClient implementation: device lifecycle for Tenstorrent Blackhole.
//
// Phase 1: Opens ttnn device 0 on client creation, closes on destruction.
// Reports device metadata (name, kind, memory) for jax.devices() discovery.
//
// Design decision: We open the device eagerly in Client_Create rather than
// lazily on first use. This matches how our Python interpreter works and
// ensures we fail fast if the device is unavailable.

#include "client.h"
#include <cstring>
#include <iostream>

// ttnn headers -- only included in .cc files, never in headers.
// This keeps compile times fast and avoids exposing ttnn types.
// TODO: Uncomment when building on remote host with ttnn available.
// #include <ttnn/ttnn.hpp>

// ============================================================
// Error helpers
// ============================================================

PJRT_Error* MakeError(PJRT_Error_Code code, const std::string& message) {
  auto* err = new PJRT_Error;
  err->code = code;
  err->message = message;
  return err;
}

PJRT_Event* MakeReadyEvent() {
  auto* event = new PJRT_Event;
  event->ready = true;
  event->error = nullptr;
  return event;
}

// ============================================================
// Client creation: open ttnn device 0
// ============================================================

PJRT_Error* TtClientCreate(PJRT_Client_Create_Args* args) {
  auto* client = new PJRT_Client;

  // Platform metadata
  client->platform_name = "tt";
  client->platform_version = "0.1.0-phase1";  // TODO: query ttnn version

  // Open ttnn device 0
  // TODO: Uncomment on remote host:
  // try {
  //   auto* device = &ttnn::open_device(0);
  //   client->tt_device = static_cast<void*>(device);
  // } catch (const std::exception& e) {
  //   delete client;
  //   return MakeError(PJRT_Error_Code_INTERNAL,
  //       std::string("Failed to open ttnn device 0: ") + e.what());
  // }
  client->tt_device = nullptr;  // Placeholder until ttnn is linked

  // Set up device description
  client->device.description.id = 0;
  client->device.description.kind = "Blackhole";
  client->device.description.debug_str = "TT Blackhole P150 (device 0)";
  client->device.description.to_string = "TT Blackhole P150 (device 0)";

  // Device metadata
  client->device.local_hardware_id = 0;
  client->device.is_addressable = true;

  // Memory space: DRAM (Blackhole has ~12GB GDDR6)
  client->device.dram_memory.id = 0;
  client->device.dram_memory.kind = "device";
  client->device.dram_memory.debug_str = "TT DRAM (device 0)";
  client->device.dram_memory.to_string = "TT DRAM (device 0)";
  client->device.dram_memory.device_ptr =
      reinterpret_cast<PJRT_Device*>(&client->device);

  // Wire up memory pointers
  client->memory_obj.inner = &client->device.dram_memory;
  client->device.memory_ptrs.push_back(
      reinterpret_cast<PJRT_Memory*>(&client->memory_obj));
  client->device.default_memory_ptr =
      reinterpret_cast<PJRT_Memory*>(&client->memory_obj);

  // Wire up device description pointer
  client->device_desc_obj.inner = &client->device.description;

  // Stable pointer array for device enumeration
  client->device_ptrs.push_back(&client->device);

  args->client = client;
  return nullptr;  // Success
}

// ============================================================
// Client destruction: close ttnn device
// ============================================================

PJRT_Error* TtClientDestroy(PJRT_Client_Destroy_Args* args) {
  auto* client = args->client;
  if (!client) return nullptr;

  // Close ttnn device
  // TODO: Uncomment on remote host:
  // if (client->tt_device) {
  //   try {
  //     ttnn::close_device(*static_cast<ttnn::Device*>(client->tt_device));
  //   } catch (const std::exception& e) {
  //     std::cerr << "Warning: failed to close ttnn device: "
  //               << e.what() << std::endl;
  //   }
  // }

  delete client;
  return nullptr;
}
