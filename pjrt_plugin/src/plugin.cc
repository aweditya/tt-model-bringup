// PJRT Plugin Entry Point for Tenstorrent Blackhole
//
// Implements the PJRT C API function pointer table (PJRT_Api) and the
// GetPjrtApi() entry point that JAX calls after dlopen.
//
// The real pjrt_c_api.h from XLA (jaxlib 0.6.2, PJRT API v0.70) has 115
// function pointers. We implement the minimum set for Phase 1 (device
// discovery) and set everything else to nullptr. JAX gracefully handles
// nullptr function pointers by reporting "not implemented."

#include "pjrt_c_api.h"
#include "client.h"

#include <cstring>
#include <iostream>

// ============================================================
// Error API
// ============================================================

static void ErrorDestroy(PJRT_Error_Destroy_Args* args) {
  delete args->error;
}

static void ErrorMessage(PJRT_Error_Message_Args* args) {
  if (args->error) {
    args->message = args->error->message.c_str();
    args->message_size = args->error->message.size();
  } else {
    args->message = "";
    args->message_size = 0;
  }
}

static PJRT_Error* ErrorGetCode(PJRT_Error_GetCode_Args* args) {
  if (args->error) {
    args->code = args->error->code;
  } else {
    args->code = PJRT_Error_Code_INTERNAL;
  }
  return nullptr;
}

// ============================================================
// Plugin API
// ============================================================

static PJRT_Error* PluginInitialize(PJRT_Plugin_Initialize_Args* args) {
  return nullptr;
}

static PJRT_Error* PluginAttributes(PJRT_Plugin_Attributes_Args* args) {
  args->num_attributes = 0;
  args->attributes = nullptr;
  return nullptr;
}

// ============================================================
// Event API (synchronous Phase 1 — always immediately ready)
// ============================================================

static PJRT_Error* EventDestroy(PJRT_Event_Destroy_Args* args) {
  if (args->event) {
    delete args->event->error;
    delete args->event;
  }
  return nullptr;
}

static PJRT_Error* EventIsReady(PJRT_Event_IsReady_Args* args) {
  args->is_ready = true;
  return nullptr;
}

static PJRT_Error* EventError(PJRT_Event_Error_Args* args) {
  // Returns nullptr (no error) for synchronous events.
  return nullptr;
}

static PJRT_Error* EventAwait(PJRT_Event_Await_Args* args) {
  return nullptr;  // Nothing to wait for.
}

static PJRT_Error* EventOnReady(PJRT_Event_OnReady_Args* args) {
  if (args->callback) {
    args->callback(nullptr, args->user_arg);
  }
  return nullptr;
}

// ============================================================
// Client API
// ============================================================

static PJRT_Error* ClientPlatformName(PJRT_Client_PlatformName_Args* args) {
  args->platform_name = args->client->platform_name.c_str();
  args->platform_name_size = args->client->platform_name.size();
  return nullptr;
}

static PJRT_Error* ClientProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  args->process_index = 0;
  return nullptr;
}

static PJRT_Error* ClientPlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  args->platform_version = args->client->platform_version.c_str();
  args->platform_version_size = args->client->platform_version.size();
  return nullptr;
}

static PJRT_Error* ClientDevices(PJRT_Client_Devices_Args* args) {
  args->devices = args->client->device_ptrs.data();
  args->num_devices = args->client->device_ptrs.size();
  return nullptr;
}

static PJRT_Error* ClientAddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  args->addressable_devices = args->client->device_ptrs.data();
  args->num_addressable_devices = args->client->device_ptrs.size();
  return nullptr;
}

static PJRT_Error* ClientLookupDevice(PJRT_Client_LookupDevice_Args* args) {
  if (args->id == 0) {
    args->device = &args->client->device;
    return nullptr;
  }
  return MakeError(PJRT_Error_Code_NOT_FOUND, "Device not found");
}

static PJRT_Error* ClientLookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  if (args->local_hardware_id == 0) {
    args->addressable_device = &args->client->device;
    return nullptr;
  }
  return MakeError(PJRT_Error_Code_NOT_FOUND, "Addressable device not found");
}

static PJRT_Error* ClientAddressableMemories(
    PJRT_Client_AddressableMemories_Args* args) {
  args->addressable_memories = args->client->memory_ptrs.data();
  args->num_addressable_memories = args->client->memory_ptrs.size();
  return nullptr;
}

// ============================================================
// Device API
// ============================================================

static PJRT_Error* DeviceGetDescription(
    PJRT_Device_GetDescription_Args* args) {
  args->device_description = &args->device->description;
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

static PJRT_Error* DeviceAddressableMemories(
    PJRT_Device_AddressableMemories_Args* args) {
  args->memories = args->device->memory_ptrs.data();
  args->num_memories = args->device->memory_ptrs.size();
  return nullptr;
}

static PJRT_Error* DeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  args->memory = args->device->default_memory_ptr;
  return nullptr;
}

// ============================================================
// DeviceDescription API
// ============================================================

static PJRT_Error* DeviceDescriptionId(
    PJRT_DeviceDescription_Id_Args* args) {
  args->id = args->device_description->id;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  args->process_index = 0;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionAttributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionKind(
    PJRT_DeviceDescription_Kind_Args* args) {
  args->device_kind = args->device_description->kind.c_str();
  args->device_kind_size = args->device_description->kind.size();
  return nullptr;
}

static PJRT_Error* DeviceDescriptionDebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  args->debug_string = args->device_description->debug_str.c_str();
  args->debug_string_size = args->device_description->debug_str.size();
  return nullptr;
}

static PJRT_Error* DeviceDescriptionToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  args->to_string = args->device_description->to_string.c_str();
  args->to_string_size = args->device_description->to_string.size();
  return nullptr;
}

// ============================================================
// Memory API
// ============================================================

static PJRT_Error* MemoryId(PJRT_Memory_Id_Args* args) {
  args->id = args->memory->id;
  return nullptr;
}

static PJRT_Error* MemoryKind(PJRT_Memory_Kind_Args* args) {
  args->kind = args->memory->kind.c_str();
  args->kind_size = args->memory->kind.size();
  return nullptr;
}

static PJRT_Error* MemoryDebugString(PJRT_Memory_DebugString_Args* args) {
  args->debug_string = args->memory->debug_str.c_str();
  args->debug_string_size = args->memory->debug_str.size();
  return nullptr;
}

static PJRT_Error* MemoryToString(PJRT_Memory_ToString_Args* args) {
  args->to_string = args->memory->to_string.c_str();
  args->to_string_size = args->memory->to_string.size();
  return nullptr;
}

static PJRT_Error* MemoryAddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  static PJRT_Device* dev_arr[1];
  dev_arr[0] = args->memory->device_ptr;
  args->devices = dev_arr;
  args->num_devices = 1;
  return nullptr;
}

static PJRT_Error* MemoryKindId(PJRT_Memory_Kind_Id_Args* args) {
  args->kind_id = args->memory->kind_id;
  return nullptr;
}

// ============================================================
// Buffer API (Phase 2 stubs)
// ============================================================

static PJRT_Error* BufferDestroy(PJRT_Buffer_Destroy_Args* args) {
  delete args->buffer;
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
  args->unpadded_dims = args->buffer->dims.data();
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

static PJRT_Error* BufferDelete(PJRT_Buffer_Delete_Args* args) {
  args->buffer->deleted = true;
  return nullptr;
}

static PJRT_Error* BufferIsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  args->is_deleted = args->buffer->deleted;
  return nullptr;
}

static PJRT_Error* BufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  args->is_on_cpu = false;
  return nullptr;
}

static PJRT_Error* BufferReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  args->event = MakeReadyEvent();
  return nullptr;
}

// ============================================================
// Executable API (Phase 3 stubs)
// ============================================================

static PJRT_Error* ExecutableDestroy(PJRT_Executable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

static PJRT_Error* ExecutableName(PJRT_Executable_Name_Args* args) {
  args->executable_name = args->executable->name.c_str();
  args->executable_name_size = args->executable->name.size();
  return nullptr;
}

static PJRT_Error* ExecutableNumOutputs(
    PJRT_Executable_NumOutputs_Args* args) {
  args->num_outputs = args->executable->num_outputs;
  return nullptr;
}

static PJRT_Error* ExecutableSizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  args->size_in_bytes = 0;
  return nullptr;
}

// ============================================================
// LoadedExecutable API (Phase 3 stubs)
// ============================================================

static PJRT_Error* LoadedExecutableDestroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

static PJRT_Error* LoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  auto* exec = new PJRT_Executable;
  exec->name = args->loaded_executable->executable.name;
  exec->num_outputs = args->loaded_executable->executable.num_outputs;
  args->executable = exec;
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

// ============================================================
// Generic UNIMPLEMENTED stubs for functions we haven't built yet.
// These return UNIMPLEMENTED error instead of crashing on nullptr.
// ============================================================

// Generic UNIMPLEMENTED stub. Since all PJRT function types take a single
// pointer arg and return PJRT_Error*, we use one function and cast it.
static PJRT_Error* Unimplemented(void* args) {
  return MakeError(PJRT_Error_Code_UNIMPLEMENTED, "Not yet implemented");
}

// ============================================================
// PJRT_Api function pointer table
//
// Must match the exact field order in the real pjrt_c_api.h.
// ALL function pointers are set — no nullptrs.
// ============================================================

// Cast the Unimplemented stub to the expected function pointer type.
// field_name is the PJRT_Api field name (e.g., PJRT_Client_Compile).
#define STUB(field_name) reinterpret_cast<decltype(api.field_name)>(Unimplemented)

static PJRT_Api BuildApi() {
  PJRT_Api api;
  memset(&api, 0, sizeof(api));

  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.extension_start = nullptr;

  // Version
  api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  api.pjrt_api_version.extension_start = nullptr;
  api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  api.pjrt_api_version.minor_version = PJRT_API_MINOR;

  // Error
  api.PJRT_Error_Destroy = ErrorDestroy;
  api.PJRT_Error_Message = ErrorMessage;
  api.PJRT_Error_GetCode = ErrorGetCode;

  // Plugin
  api.PJRT_Plugin_Initialize = PluginInitialize;
  api.PJRT_Plugin_Attributes = PluginAttributes;

  // Event
  api.PJRT_Event_Destroy = EventDestroy;
  api.PJRT_Event_IsReady = EventIsReady;
  api.PJRT_Event_Error = EventError;
  api.PJRT_Event_Await = EventAwait;
  api.PJRT_Event_OnReady = EventOnReady;

  // Client
  api.PJRT_Client_Create = TtClientCreate;
  api.PJRT_Client_Destroy = TtClientDestroy;
  api.PJRT_Client_PlatformName = ClientPlatformName;
  api.PJRT_Client_ProcessIndex = ClientProcessIndex;
  api.PJRT_Client_PlatformVersion = ClientPlatformVersion;
  api.PJRT_Client_Devices = ClientDevices;
  api.PJRT_Client_AddressableDevices = ClientAddressableDevices;
  api.PJRT_Client_LookupDevice = ClientLookupDevice;
  api.PJRT_Client_LookupAddressableDevice = ClientLookupAddressableDevice;
  api.PJRT_Client_AddressableMemories = ClientAddressableMemories;
  api.PJRT_Client_Compile = STUB(PJRT_Client_Compile);
  api.PJRT_Client_DefaultDeviceAssignment = STUB(PJRT_Client_DefaultDeviceAssignment);
  api.PJRT_Client_BufferFromHostBuffer = STUB(PJRT_Client_BufferFromHostBuffer);

  // DeviceDescription
  api.PJRT_DeviceDescription_Id = DeviceDescriptionId;
  api.PJRT_DeviceDescription_ProcessIndex = DeviceDescriptionProcessIndex;
  api.PJRT_DeviceDescription_Attributes = DeviceDescriptionAttributes;
  api.PJRT_DeviceDescription_Kind = DeviceDescriptionKind;
  api.PJRT_DeviceDescription_DebugString = DeviceDescriptionDebugString;
  api.PJRT_DeviceDescription_ToString = DeviceDescriptionToString;

  // Device
  api.PJRT_Device_GetDescription = DeviceGetDescription;
  api.PJRT_Device_IsAddressable = DeviceIsAddressable;
  api.PJRT_Device_LocalHardwareId = DeviceLocalHardwareId;
  api.PJRT_Device_AddressableMemories = DeviceAddressableMemories;
  api.PJRT_Device_DefaultMemory = DeviceDefaultMemory;
  api.PJRT_Device_MemoryStats = STUB(PJRT_Device_MemoryStats);

  // Memory
  api.PJRT_Memory_Id = MemoryId;
  api.PJRT_Memory_Kind = MemoryKind;
  api.PJRT_Memory_DebugString = MemoryDebugString;
  api.PJRT_Memory_ToString = MemoryToString;
  api.PJRT_Memory_AddressableByDevices = MemoryAddressableByDevices;

  // Executable
  api.PJRT_Executable_Destroy = ExecutableDestroy;
  api.PJRT_Executable_Name = ExecutableName;
  api.PJRT_Executable_NumReplicas = STUB(PJRT_Executable_NumReplicas);
  api.PJRT_Executable_NumPartitions = STUB(PJRT_Executable_NumPartitions);
  api.PJRT_Executable_NumOutputs = ExecutableNumOutputs;
  api.PJRT_Executable_SizeOfGeneratedCodeInBytes =
      ExecutableSizeOfGeneratedCodeInBytes;
  api.PJRT_Executable_GetCostAnalysis = STUB(PJRT_Executable_GetCostAnalysis);
  api.PJRT_Executable_OutputMemoryKinds = STUB(PJRT_Executable_OutputMemoryKinds);
  api.PJRT_Executable_OptimizedProgram = STUB(PJRT_Executable_OptimizedProgram);
  api.PJRT_Executable_Serialize = STUB(PJRT_Executable_Serialize);

  // LoadedExecutable
  api.PJRT_LoadedExecutable_Destroy = LoadedExecutableDestroy;
  api.PJRT_LoadedExecutable_GetExecutable = LoadedExecutableGetExecutable;
  api.PJRT_LoadedExecutable_AddressableDevices =
      LoadedExecutableAddressableDevices;
  api.PJRT_LoadedExecutable_Delete = STUB(PJRT_LoadedExecutable_Delete);
  api.PJRT_LoadedExecutable_IsDeleted = STUB(PJRT_LoadedExecutable_IsDeleted);
  api.PJRT_LoadedExecutable_Execute = STUB(PJRT_LoadedExecutable_Execute);
  api.PJRT_Executable_DeserializeAndLoad = STUB(PJRT_Executable_DeserializeAndLoad);
  api.PJRT_LoadedExecutable_Fingerprint = STUB(PJRT_LoadedExecutable_Fingerprint);

  // Buffer
  api.PJRT_Buffer_Destroy = BufferDestroy;
  api.PJRT_Buffer_ElementType = BufferElementType;
  api.PJRT_Buffer_Dimensions = BufferDimensions;
  api.PJRT_Buffer_UnpaddedDimensions = BufferUnpaddedDimensions;
  api.PJRT_Buffer_DynamicDimensionIndices = STUB(PJRT_Buffer_DynamicDimensionIndices);
  api.PJRT_Buffer_GetMemoryLayout = STUB(PJRT_Buffer_GetMemoryLayout);
  api.PJRT_Buffer_OnDeviceSizeInBytes = BufferOnDeviceSizeInBytes;
  api.PJRT_Buffer_Device = BufferDevice;
  api.PJRT_Buffer_Memory = BufferMemory;
  api.PJRT_Buffer_Delete = BufferDelete;
  api.PJRT_Buffer_IsDeleted = BufferIsDeleted;
  api.PJRT_Buffer_CopyToDevice = STUB(PJRT_Buffer_CopyToDevice);
  api.PJRT_Buffer_ToHostBuffer = STUB(PJRT_Buffer_ToHostBuffer);
  api.PJRT_Buffer_IsOnCpu = BufferIsOnCpu;
  api.PJRT_Buffer_ReadyEvent = BufferReadyEvent;
  api.PJRT_Buffer_UnsafePointer = STUB(PJRT_Buffer_UnsafePointer);
  api.PJRT_Buffer_IncreaseExternalReferenceCount = STUB(PJRT_Buffer_IncreaseExternalReferenceCount);
  api.PJRT_Buffer_DecreaseExternalReferenceCount = STUB(PJRT_Buffer_DecreaseExternalReferenceCount);
  api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = STUB(PJRT_Buffer_OpaqueDeviceMemoryDataPointer);

  // CopyToDeviceStream
  api.PJRT_CopyToDeviceStream_Destroy = STUB(PJRT_CopyToDeviceStream_Destroy);
  api.PJRT_CopyToDeviceStream_AddChunk = STUB(PJRT_CopyToDeviceStream_AddChunk);
  api.PJRT_CopyToDeviceStream_TotalBytes = STUB(PJRT_CopyToDeviceStream_TotalBytes);
  api.PJRT_CopyToDeviceStream_GranuleSize = STUB(PJRT_CopyToDeviceStream_GranuleSize);
  api.PJRT_CopyToDeviceStream_CurrentBytes = STUB(PJRT_CopyToDeviceStream_CurrentBytes);

  // TopologyDescription
  api.PJRT_TopologyDescription_Create = STUB(PJRT_TopologyDescription_Create);
  api.PJRT_TopologyDescription_Destroy = STUB(PJRT_TopologyDescription_Destroy);
  api.PJRT_TopologyDescription_PlatformName = STUB(PJRT_TopologyDescription_PlatformName);
  api.PJRT_TopologyDescription_PlatformVersion = STUB(PJRT_TopologyDescription_PlatformVersion);
  api.PJRT_TopologyDescription_GetDeviceDescriptions = STUB(PJRT_TopologyDescription_GetDeviceDescriptions);
  api.PJRT_TopologyDescription_Serialize = STUB(PJRT_TopologyDescription_Serialize);
  api.PJRT_TopologyDescription_Attributes = STUB(PJRT_TopologyDescription_Attributes);

  // Compile (standalone)
  api.PJRT_Compile = STUB(PJRT_Compile);

  // Extension fields (added after initial version)
  api.PJRT_Executable_OutputElementTypes = STUB(PJRT_Executable_OutputElementTypes);
  api.PJRT_Executable_OutputDimensions = STUB(PJRT_Executable_OutputDimensions);
  api.PJRT_Buffer_CopyToMemory = STUB(PJRT_Buffer_CopyToMemory);
  api.PJRT_Client_CreateViewOfDeviceBuffer = STUB(PJRT_Client_CreateViewOfDeviceBuffer);
  api.PJRT_Executable_Fingerprint = STUB(PJRT_Executable_Fingerprint);
  api.PJRT_Client_TopologyDescription = STUB(PJRT_Client_TopologyDescription);
  api.PJRT_Executable_GetCompiledMemoryStats = STUB(PJRT_Executable_GetCompiledMemoryStats);
  api.PJRT_Memory_Kind_Id = MemoryKindId;
  api.PJRT_ExecuteContext_Create = STUB(PJRT_ExecuteContext_Create);
  api.PJRT_ExecuteContext_Destroy = STUB(PJRT_ExecuteContext_Destroy);
  api.PJRT_Buffer_CopyRawToHost = STUB(PJRT_Buffer_CopyRawToHost);
  api.PJRT_AsyncHostToDeviceTransferManager_Destroy = STUB(PJRT_AsyncHostToDeviceTransferManager_Destroy);
  api.PJRT_AsyncHostToDeviceTransferManager_TransferData = STUB(PJRT_AsyncHostToDeviceTransferManager_TransferData);
  api.PJRT_Client_CreateBuffersForAsyncHostToDevice = STUB(PJRT_Client_CreateBuffersForAsyncHostToDevice);
  api.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = STUB(PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer);
  api.PJRT_AsyncHostToDeviceTransferManager_Device = STUB(PJRT_AsyncHostToDeviceTransferManager_Device);
  api.PJRT_AsyncHostToDeviceTransferManager_BufferCount = STUB(PJRT_AsyncHostToDeviceTransferManager_BufferCount);
  api.PJRT_AsyncHostToDeviceTransferManager_BufferSize = STUB(PJRT_AsyncHostToDeviceTransferManager_BufferSize);
  api.PJRT_AsyncHostToDeviceTransferManager_SetBufferError = STUB(PJRT_AsyncHostToDeviceTransferManager_SetBufferError);
  api.PJRT_AsyncHostToDeviceTransferManager_AddMetadata = STUB(PJRT_AsyncHostToDeviceTransferManager_AddMetadata);
  api.PJRT_Client_DmaMap = STUB(PJRT_Client_DmaMap);
  api.PJRT_Client_DmaUnmap = STUB(PJRT_Client_DmaUnmap);
  api.PJRT_Client_CreateUninitializedBuffer = STUB(PJRT_Client_CreateUninitializedBuffer);

  return api;
}

#undef STUB

static const PJRT_Api kPjrtApi = BuildApi();

// ============================================================
// Entry point: JAX dlsym's this after dlopen
// ============================================================

extern "C" {
  const PJRT_Api* GetPjrtApi() {
    return &kPjrtApi;
  }
}
