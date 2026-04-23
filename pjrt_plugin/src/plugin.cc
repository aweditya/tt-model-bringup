// PJRT Plugin Entry Point for Tenstorrent Blackhole
//
// This file defines the PJRT_Api function pointer table and the GetPjrtApi()
// entry point that JAX calls after dlopen'ing our shared library.
//
// Architecture:
// - JAX calls GetPjrtApi() -> gets PJRT_Api* with all function pointers
// - JAX calls PJRT_Client_Create -> we open ttnn device 0
// - JAX calls PJRT_Client_Devices -> we return our single Blackhole device
// - JAX calls PJRT_Client_Compile -> (Phase 3) we parse StableHLO
// - JAX calls PJRT_LoadedExecutable_Execute -> (Phase 3) we dispatch to ttnn
//
// Phase 1 implements: Error, Plugin, Client, Device, DeviceDescription,
// Memory, and Event functions. Buffer/Executable/Execute are stubbed.

#include "pjrt_c_api.h"
#include "client.h"
#include "buffer.h"
#include "executable.h"

#include <cstring>
#include <iostream>

// ============================================================
// Error API
// ============================================================

static PJRT_Error* ErrorDestroy(PJRT_Error_Destroy_Args* args) {
  delete args->error;
  return nullptr;
}

// Note: PJRT_Error_Message returns void, not PJRT_Error*.
// This is a special case in the PJRT API.
static void ErrorMessage(PJRT_Error_Message_Args* args) {
  auto* err = args->error;
  if (err) {
    args->message = err->message.c_str();
    args->message_size = err->message.size();
  } else {
    args->message = "";
    args->message_size = 0;
  }
}

static PJRT_Error* ErrorGetCode(PJRT_Error_GetCode_Args* args) {
  auto* err = args->error;
  if (err) {
    args->code = err->code;
  } else {
    args->code = PJRT_Error_Code_INTERNAL;
  }
  return nullptr;
}

// ============================================================
// Plugin API
// ============================================================

static PJRT_Error* PluginInitialize(PJRT_Plugin_Initialize_Args* args) {
  // Nothing to do at plugin init time. Device is opened in Client_Create.
  return nullptr;
}

static PJRT_Error* PluginAttributes(PJRT_Plugin_Attributes_Args* args) {
  // No custom attributes for now.
  args->num_attributes = 0;
  args->attributes = nullptr;
  return nullptr;
}

// ============================================================
// Client API
// ============================================================

static PJRT_Error* ClientPlatformName(PJRT_Client_PlatformName_Args* args) {
  auto* client = args->client;
  args->platform_name = client->platform_name.c_str();
  args->platform_name_size = client->platform_name.size();
  return nullptr;
}

static PJRT_Error* ClientProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  // Single-process, single-device setup.
  args->process_index = 0;
  return nullptr;
}

static PJRT_Error* ClientPlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  auto* client = args->client;
  args->platform_version = client->platform_version.c_str();
  args->platform_version_size = client->platform_version.size();
  return nullptr;
}

static PJRT_Error* ClientDevices(PJRT_Client_Devices_Args* args) {
  auto* client = args->client;
  args->devices = client->device_ptrs.data();
  args->num_devices = client->device_ptrs.size();
  return nullptr;
}

static PJRT_Error* ClientAddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  // Same as Devices -- single device is always addressable.
  auto* client = args->client;
  args->addressable_devices = client->device_ptrs.data();
  args->num_addressable_devices = client->device_ptrs.size();
  return nullptr;
}

// ============================================================
// Device API
// ============================================================

static PJRT_Error* DeviceGetDescription(
    PJRT_Device_GetDescription_Args* args) {
  // Return pointer to the device description embedded in the client.
  // We store a PJRT_DeviceDescription object in the client that points
  // to the description data inside the PJRT_Device.
  auto* dev = args->device;
  // We need a stable PJRT_DeviceDescription*. We store one in the client.
  // For now, use a static cast -- the client owns both the device and desc.
  // This works because we have exactly one device.
  //
  // Ugly but correct: the PJRT_DeviceDescription is allocated inside
  // PJRT_Client and its inner pointer points to dev->description.
  // We recover the client from the device by walking back.
  // TODO: Add a back-pointer from device to client for cleanliness.
  static PJRT_DeviceDescription desc;
  desc.inner = &dev->description;
  args->device_description = &desc;
  return nullptr;
}

static PJRT_Error* DeviceIsAddressable(PJRT_Device_IsAddressable_Args* args) {
  args->is_addressable = args->device->is_addressable;
  return nullptr;
}

static PJRT_Error* DeviceLocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  args->local_hardware_id = args->device->local_hardware_id;
  return nullptr;
}

static PJRT_Error* DeviceMemorySpaces(PJRT_Device_MemorySpaces_Args* args) {
  auto* dev = args->device;
  args->memory_spaces = dev->memory_ptrs.data();
  args->num_memory_spaces = dev->memory_ptrs.size();
  return nullptr;
}

static PJRT_Error* DeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  args->default_memory = args->device->default_memory_ptr;
  return nullptr;
}

// ============================================================
// DeviceDescription API
// ============================================================

static PJRT_Error* DeviceDescriptionId(
    PJRT_DeviceDescription_Id_Args* args) {
  args->id = args->device_description->inner->id;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  args->process_index = 0;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionKind(
    PJRT_DeviceDescription_Kind_Args* args) {
  auto* desc = args->device_description->inner;
  args->device_kind = desc->kind.c_str();
  args->device_kind_size = desc->kind.size();
  return nullptr;
}

static PJRT_Error* DeviceDescriptionDebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  auto* desc = args->device_description->inner;
  args->debug_string = desc->debug_str.c_str();
  args->debug_string_size = desc->debug_str.size();
  return nullptr;
}

static PJRT_Error* DeviceDescriptionToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  auto* desc = args->device_description->inner;
  args->to_string = desc->to_string.c_str();
  args->to_string_size = desc->to_string.size();
  return nullptr;
}

static PJRT_Error* DeviceDescriptionAttributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  // No custom device attributes for now.
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

// ============================================================
// Memory API
// ============================================================

static PJRT_Error* MemoryId(PJRT_Memory_Id_Args* args) {
  args->id = args->memory->inner->id;
  return nullptr;
}

static PJRT_Error* MemoryKind(PJRT_Memory_Kind_Args* args) {
  auto* mem = args->memory->inner;
  args->kind = mem->kind.c_str();
  args->kind_size = mem->kind.size();
  return nullptr;
}

static PJRT_Error* MemoryDebugString(PJRT_Memory_DebugString_Args* args) {
  auto* mem = args->memory->inner;
  args->debug_string = mem->debug_str.c_str();
  args->debug_string_size = mem->debug_str.size();
  return nullptr;
}

static PJRT_Error* MemoryToString(PJRT_Memory_ToString_Args* args) {
  auto* mem = args->memory->inner;
  args->to_string = mem->to_string.c_str();
  args->to_string_size = mem->to_string.size();
  return nullptr;
}

static PJRT_Error* MemoryAddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  auto* mem = args->memory->inner;
  // The memory's device_ptr points to the PJRT_Device that owns it.
  // We need to return an array of PJRT_Device*.
  static PJRT_Device* dev_ptr;
  dev_ptr = mem->device_ptr;
  args->devices = &dev_ptr;
  args->num_devices = 1;
  return nullptr;
}

// ============================================================
// Buffer API (Phase 1: stubs)
// ============================================================

static PJRT_Error* BufferDestroy(PJRT_Buffer_Destroy_Args* args) {
  if (args->buffer) {
    args->buffer->deleted = true;
    // TODO Phase 2: deallocate ttnn tensor
    delete args->buffer;
  }
  return nullptr;
}

static PJRT_Error* BufferElementType(PJRT_Buffer_ElementType_Args* args) {
  args->type = args->buffer->element_type;
  return nullptr;
}

static PJRT_Error* BufferDimensions(PJRT_Buffer_Dimensions_Args* args) {
  args->dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

static PJRT_Error* BufferUnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args) {
  // No padding -- unpadded dims are the same as dims.
  args->dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

static PJRT_Error* BufferOnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  args->on_device_size_in_bytes = args->buffer->size_bytes;
  return nullptr;
}

static PJRT_Error* BufferDevice(PJRT_Buffer_Device_Args* args) {
  args->device = args->buffer->device;
  return nullptr;
}

static PJRT_Error* BufferMemory(PJRT_Buffer_Memory_Args* args) {
  args->memory = args->buffer->memory;
  return nullptr;
}

static PJRT_Error* BufferIsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  args->is_deleted = args->buffer->deleted;
  return nullptr;
}

static PJRT_Error* BufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  return MakeError(PJRT_Error_Code_UNIMPLEMENTED,
                   "PJRT_Buffer_ToHostBuffer not yet implemented (Phase 2)");
}

static PJRT_Error* BufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  args->is_on_cpu = false;  // Our buffers are always on device.
  return nullptr;
}

static PJRT_Error* BufferReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  // Phase 1: synchronous, so buffers are always ready.
  args->event = MakeReadyEvent();
  return nullptr;
}

// ============================================================
// Executable API (Phase 1: stubs)
// ============================================================

static PJRT_Error* ExecutableDestroy(PJRT_Executable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

static PJRT_Error* ExecutableName(PJRT_Executable_Name_Args* args) {
  args->name = args->executable->name.c_str();
  args->name_size = args->executable->name.size();
  return nullptr;
}

static PJRT_Error* ExecutableNumOutputs(
    PJRT_Executable_NumOutputs_Args* args) {
  args->num_outputs = args->executable->num_outputs;
  return nullptr;
}

static PJRT_Error* ExecutableSizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  args->size_in_bytes = 0;  // Interpretation, not compilation.
  return nullptr;
}

static PJRT_Error* ExecutableFingerprint(
    PJRT_Executable_Fingerprint_Args* args) {
  args->fingerprint = nullptr;
  args->fingerprint_size = 0;
  return nullptr;
}

// ============================================================
// LoadedExecutable API (Phase 1: stubs)
// ============================================================

static PJRT_Error* LoadedExecutableDestroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

static PJRT_Error* LoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  // Return a copy of the unloaded executable metadata.
  auto* exec = new PJRT_Executable;
  exec->name = args->executable->executable.name;
  exec->num_outputs = args->executable->executable.num_outputs;
  args->unloaded_executable = exec;
  return nullptr;
}

static PJRT_Error* LoadedExecutableAddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  args->addressable_devices =
      args->executable->addressable_device_ptrs.data();
  args->num_addressable_devices =
      args->executable->addressable_device_ptrs.size();
  return nullptr;
}

static PJRT_Error* ClientCompile(PJRT_Client_Compile_Args* args) {
  return MakeError(PJRT_Error_Code_UNIMPLEMENTED,
                   "PJRT_Client_Compile not yet implemented (Phase 3)");
}

static PJRT_Error* ClientBufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  return MakeError(PJRT_Error_Code_UNIMPLEMENTED,
      "PJRT_Client_BufferFromHostBuffer not yet implemented (Phase 2)");
}

static PJRT_Error* LoadedExecutableExecute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  return MakeError(PJRT_Error_Code_UNIMPLEMENTED,
                   "PJRT_LoadedExecutable_Execute not yet implemented (Phase 3)");
}

// ============================================================
// Event API (synchronous -- events are always immediately ready)
// ============================================================

static PJRT_Error* EventDestroy(PJRT_Event_Destroy_Args* args) {
  if (args->event) {
    delete args->event->error;  // May be nullptr, that's fine.
    delete args->event;
  }
  return nullptr;
}

static PJRT_Error* EventIsReady(PJRT_Event_IsReady_Args* args) {
  args->is_ready = true;  // Synchronous: always ready.
  return nullptr;
}

static PJRT_Error* EventError(PJRT_Event_Error_Args* args) {
  args->error = args->event ? args->event->error : nullptr;
  return nullptr;
}

static PJRT_Error* EventAwait(PJRT_Event_Await_Args* args) {
  // Synchronous: nothing to wait for.
  return nullptr;
}

static PJRT_Error* EventOnReady(PJRT_Event_OnReady_Args* args) {
  // Synchronous: invoke callback immediately.
  if (args->callback) {
    args->callback(nullptr, args->user_arg);
  }
  return nullptr;
}

// ============================================================
// PJRT_Api function pointer table
// ============================================================

static const PJRT_Api kPjrtApi = {
    .struct_size = sizeof(PJRT_Api),
    .extension_start = nullptr,

    // PJRT API version. Must match what jaxlib expects.
    // TODO: Query the exact version from installed jaxlib.
    .pjrt_api_version_major = 0,
    .pjrt_api_version_minor = 54,

    // Error
    .PJRT_Error_Destroy = ErrorDestroy,
    .PJRT_Error_Message = ErrorMessage,
    .PJRT_Error_GetCode = ErrorGetCode,

    // Plugin
    .PJRT_Plugin_Initialize = PluginInitialize,
    .PJRT_Plugin_Attributes = PluginAttributes,

    // Client
    .PJRT_Client_Create = TtClientCreate,
    .PJRT_Client_Destroy = TtClientDestroy,
    .PJRT_Client_PlatformName = ClientPlatformName,
    .PJRT_Client_ProcessIndex = ClientProcessIndex,
    .PJRT_Client_PlatformVersion = ClientPlatformVersion,
    .PJRT_Client_Devices = ClientDevices,
    .PJRT_Client_AddressableDevices = ClientAddressableDevices,
    .PJRT_Client_Compile = ClientCompile,
    .PJRT_Client_BufferFromHostBuffer = ClientBufferFromHostBuffer,

    // DeviceDescription
    .PJRT_DeviceDescription_Id = DeviceDescriptionId,
    .PJRT_DeviceDescription_ProcessIndex = DeviceDescriptionProcessIndex,
    .PJRT_DeviceDescription_Kind = DeviceDescriptionKind,
    .PJRT_DeviceDescription_DebugString = DeviceDescriptionDebugString,
    .PJRT_DeviceDescription_ToString = DeviceDescriptionToString,
    .PJRT_DeviceDescription_Attributes = DeviceDescriptionAttributes,

    // Device
    .PJRT_Device_GetDescription = DeviceGetDescription,
    .PJRT_Device_IsAddressable = DeviceIsAddressable,
    .PJRT_Device_LocalHardwareId = DeviceLocalHardwareId,
    .PJRT_Device_MemorySpaces = DeviceMemorySpaces,
    .PJRT_Device_DefaultMemory = DeviceDefaultMemory,

    // Memory
    .PJRT_Memory_Id = MemoryId,
    .PJRT_Memory_Kind = MemoryKind,
    .PJRT_Memory_DebugString = MemoryDebugString,
    .PJRT_Memory_ToString = MemoryToString,
    .PJRT_Memory_AddressableByDevices = MemoryAddressableByDevices,

    // Buffer
    .PJRT_Buffer_Destroy = BufferDestroy,
    .PJRT_Buffer_ElementType = BufferElementType,
    .PJRT_Buffer_Dimensions = BufferDimensions,
    .PJRT_Buffer_UnpaddedDimensions = BufferUnpaddedDimensions,
    .PJRT_Buffer_OnDeviceSizeInBytes = BufferOnDeviceSizeInBytes,
    .PJRT_Buffer_Device = BufferDevice,
    .PJRT_Buffer_Memory = BufferMemory,
    .PJRT_Buffer_IsDeleted = BufferIsDeleted,
    .PJRT_Buffer_ToHostBuffer = BufferToHostBuffer,
    .PJRT_Buffer_IsOnCpu = BufferIsOnCpu,
    .PJRT_Buffer_ReadyEvent = BufferReadyEvent,

    // Executable
    .PJRT_Executable_Destroy = ExecutableDestroy,
    .PJRT_Executable_Name = ExecutableName,
    .PJRT_Executable_NumOutputs = ExecutableNumOutputs,
    .PJRT_Executable_SizeOfGeneratedCodeInBytes =
        ExecutableSizeOfGeneratedCodeInBytes,
    .PJRT_Executable_Fingerprint = ExecutableFingerprint,

    // LoadedExecutable
    .PJRT_LoadedExecutable_Destroy = LoadedExecutableDestroy,
    .PJRT_LoadedExecutable_GetExecutable = LoadedExecutableGetExecutable,
    .PJRT_LoadedExecutable_AddressableDevices =
        LoadedExecutableAddressableDevices,
    .PJRT_LoadedExecutable_Execute = LoadedExecutableExecute,

    // Event
    .PJRT_Event_Destroy = EventDestroy,
    .PJRT_Event_IsReady = EventIsReady,
    .PJRT_Event_Error = EventError,
    .PJRT_Event_Await = EventAwait,
    .PJRT_Event_OnReady = EventOnReady,
};

// ============================================================
// Entry point: JAX dlsym's this after dlopen'ing our .so
// ============================================================

extern "C" const PJRT_Api* GetPjrtApi() {
  return &kPjrtApi;
}
