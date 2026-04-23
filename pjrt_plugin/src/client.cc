// TtClient implementation: device lifecycle for Tenstorrent Blackhole.
//
// Phase 1: Opens ttnn device 0 on client creation, closes on destruction.
// Reports device metadata for jax.devices() discovery.

#include "client.h"
#include <cstring>

// TODO: Uncomment when building with ttnn linked
// #include <ttnn/ttnn.hpp>

// ============================================================
// Error / Event helpers
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
  client->platform_version = "0.1.0-phase1";

  // Open ttnn device 0
  // TODO: Uncomment on remote host with ttnn:
  // try {
  //   auto* device = &ttnn::open_device(0);
  //   client->tt_device = static_cast<void*>(device);
  // } catch (const std::exception& e) {
  //   delete client;
  //   return MakeError(PJRT_Error_Code_INTERNAL,
  //       std::string("Failed to open ttnn device 0: ") + e.what());
  // }
  client->tt_device = nullptr;

  // Device description
  client->device.description.id = 0;
  client->device.description.kind = "Blackhole";
  client->device.description.debug_str = "TT Blackhole (device 0)";
  client->device.description.to_string = "TT Blackhole (device 0)";

  // Device metadata
  client->device.local_hardware_id = 0;
  client->device.is_addressable = true;
  client->device.client = client;

  // DRAM memory space
  client->device.dram_memory.id = 0;
  client->device.dram_memory.kind = "device";
  client->device.dram_memory.kind_id = 0;
  client->device.dram_memory.debug_str = "TT DRAM (device 0)";
  client->device.dram_memory.to_string = "TT DRAM (device 0)";
  client->device.dram_memory.device_ptr = &client->device;

  // Wire up memory pointers
  client->device.memory_ptrs.push_back(&client->device.dram_memory);
  client->device.default_memory_ptr = &client->device.dram_memory;

  // Client-level pointer arrays
  client->device_ptrs.push_back(&client->device);
  client->memory_ptrs.push_back(&client->device.dram_memory);

  args->client = client;
  return nullptr;
}

// ============================================================
// Client destruction: close ttnn device
// ============================================================

PJRT_Error* TtClientDestroy(PJRT_Client_Destroy_Args* args) {
  auto* client = args->client;
  if (!client) return nullptr;

  // TODO: Uncomment on remote host:
  // if (client->tt_device) {
  //   try {
  //     ttnn::close_device(*static_cast<ttnn::Device*>(client->tt_device));
  //   } catch (...) {}
  // }

  delete client;
  return nullptr;
}
