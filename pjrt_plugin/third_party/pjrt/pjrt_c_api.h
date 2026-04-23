// PJRT C API Header - Vendored Subset for Phase 1
//
// Minimal extraction of PJRT C API types for device discovery.
// BEFORE BUILDING AGAINST REAL JAXLIB: Replace with the exact pjrt_c_api.h
// from the XLA commit matching your jaxlib version. See scripts/fetch_pjrt_header.sh.
//
// The field ORDER in PJRT_Api determines ABI compatibility. This layout is
// based on XLA's PJRT API version 0.54 (circa jaxlib 0.5.x, early 2026).

#ifndef PJRT_C_API_H_
#define PJRT_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types. Our plugin defines the concrete structs internally;
// JAX only ever holds pointers to these.
typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Event PJRT_Event;

// Error codes (matches gRPC / absl::StatusCode)
typedef enum {
  PJRT_Error_Code_CANCELLED = 1,
  PJRT_Error_Code_UNKNOWN = 2,
  PJRT_Error_Code_INVALID_ARGUMENT = 3,
  PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
  PJRT_Error_Code_NOT_FOUND = 5,
  PJRT_Error_Code_ALREADY_EXISTS = 6,
  PJRT_Error_Code_PERMISSION_DENIED = 7,
  PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
  PJRT_Error_Code_FAILED_PRECONDITION = 9,
  PJRT_Error_Code_ABORTED = 10,
  PJRT_Error_Code_OUT_OF_RANGE = 11,
  PJRT_Error_Code_UNIMPLEMENTED = 12,
  PJRT_Error_Code_INTERNAL = 13,
  PJRT_Error_Code_UNAVAILABLE = 14,
  PJRT_Error_Code_DATA_LOSS = 15,
  PJRT_Error_Code_UNAUTHENTICATED = 16,
} PJRT_Error_Code;

typedef enum {
  PJRT_Buffer_Type_INVALID = 0,
  PJRT_Buffer_Type_PRED = 1,
  PJRT_Buffer_Type_S8 = 2,
  PJRT_Buffer_Type_S16 = 3,
  PJRT_Buffer_Type_S32 = 4,
  PJRT_Buffer_Type_S64 = 5,
  PJRT_Buffer_Type_U8 = 6,
  PJRT_Buffer_Type_U16 = 7,
  PJRT_Buffer_Type_U32 = 8,
  PJRT_Buffer_Type_U64 = 9,
  PJRT_Buffer_Type_F16 = 10,
  PJRT_Buffer_Type_F32 = 11,
  PJRT_Buffer_Type_F64 = 12,
  PJRT_Buffer_Type_C64 = 15,
  PJRT_Buffer_Type_BF16 = 16,
  PJRT_Buffer_Type_TOKEN = 17,
  PJRT_Buffer_Type_C128 = 18,
} PJRT_Buffer_Type;

typedef enum {
  PJRT_HostBufferSemantics_kImmutableOnlyDuringCall = 0,
  PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes = 1,
  PJRT_HostBufferSemantics_kImmutableZeroCopy = 2,
  PJRT_HostBufferSemantics_kMutableZeroCopy = 3,
} PJRT_HostBufferSemantics;

// Key-value pairs for config / attributes
typedef enum {
  PJRT_NamedValue_kString = 0,
  PJRT_NamedValue_kInt64 = 1,
  PJRT_NamedValue_kInt64List = 2,
  PJRT_NamedValue_kFloat = 3,
  PJRT_NamedValue_kBool = 4,
} PJRT_NamedValue_Type;

typedef struct {
  size_t struct_size;
  const char* name;
  size_t name_size;
  PJRT_NamedValue_Type type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
  };
  size_t value_size;
} PJRT_NamedValue;

// ============================================================
// Argument structs -- one per PJRT function
// Input fields first, output fields after the comment "output:"
// ============================================================

// Error
typedef struct { size_t struct_size; PJRT_Error* error; } PJRT_Error_Destroy_Args;
typedef struct { size_t struct_size; const PJRT_Error* error; /*out*/ const char* message; size_t message_size; } PJRT_Error_Message_Args;
typedef struct { size_t struct_size; const PJRT_Error* error; /*out*/ PJRT_Error_Code code; } PJRT_Error_GetCode_Args;

// Plugin
typedef struct { size_t struct_size; void* extension_start; } PJRT_Plugin_Initialize_Args;
typedef struct { size_t struct_size; /*out*/ size_t num_attributes; const PJRT_NamedValue* attributes; } PJRT_Plugin_Attributes_Args;

// Client
typedef struct { size_t struct_size; const PJRT_NamedValue* create_options; size_t num_options; /*out*/ PJRT_Client* client; } PJRT_Client_Create_Args;
typedef struct { size_t struct_size; PJRT_Client* client; } PJRT_Client_Destroy_Args;
typedef struct { size_t struct_size; PJRT_Client* client; /*out*/ const char* platform_name; size_t platform_name_size; } PJRT_Client_PlatformName_Args;
typedef struct { size_t struct_size; PJRT_Client* client; /*out*/ int process_index; } PJRT_Client_ProcessIndex_Args;
typedef struct { size_t struct_size; PJRT_Client* client; /*out*/ const char* platform_version; size_t platform_version_size; } PJRT_Client_PlatformVersion_Args;
typedef struct { size_t struct_size; PJRT_Client* client; /*out*/ PJRT_Device* const* devices; size_t num_devices; } PJRT_Client_Devices_Args;
typedef struct { size_t struct_size; PJRT_Client* client; /*out*/ PJRT_Device* const* addressable_devices; size_t num_addressable_devices; } PJRT_Client_AddressableDevices_Args;

// Program (for Compile)
typedef struct { size_t struct_size; const char* code; size_t code_size; const char* format; size_t format_size; } PJRT_Program;
typedef struct { size_t struct_size; PJRT_Client* client; const PJRT_Program* program; /*out*/ PJRT_LoadedExecutable* executable; } PJRT_Client_Compile_Args;

// BufferFromHostBuffer
typedef struct {
  size_t struct_size; PJRT_Client* client;
  const void* data; PJRT_Buffer_Type type;
  const int64_t* dims; size_t num_dims;
  const int64_t* byte_strides; size_t num_byte_strides;
  PJRT_HostBufferSemantics host_buffer_semantics;
  PJRT_Device* device; PJRT_Memory* memory;
  /*out*/ PJRT_Buffer* buffer; PJRT_Event* done_with_host_buffer;
} PJRT_Client_BufferFromHostBuffer_Args;

// Device
typedef struct { size_t struct_size; PJRT_Device* device; /*out*/ PJRT_DeviceDescription* device_description; } PJRT_Device_GetDescription_Args;
typedef struct { size_t struct_size; PJRT_Device* device; /*out*/ bool is_addressable; } PJRT_Device_IsAddressable_Args;
typedef struct { size_t struct_size; PJRT_Device* device; /*out*/ int local_hardware_id; } PJRT_Device_LocalHardwareId_Args;
typedef struct { size_t struct_size; PJRT_Device* device; /*out*/ PJRT_Memory* const* memory_spaces; size_t num_memory_spaces; } PJRT_Device_MemorySpaces_Args;
typedef struct { size_t struct_size; PJRT_Device* device; /*out*/ PJRT_Memory* default_memory; } PJRT_Device_DefaultMemory_Args;

// DeviceDescription
typedef struct { size_t struct_size; PJRT_DeviceDescription* device_description; /*out*/ int id; } PJRT_DeviceDescription_Id_Args;
typedef struct { size_t struct_size; PJRT_DeviceDescription* device_description; /*out*/ int process_index; } PJRT_DeviceDescription_ProcessIndex_Args;
typedef struct { size_t struct_size; PJRT_DeviceDescription* device_description; /*out*/ const char* device_kind; size_t device_kind_size; } PJRT_DeviceDescription_Kind_Args;
typedef struct { size_t struct_size; PJRT_DeviceDescription* device_description; /*out*/ const char* debug_string; size_t debug_string_size; } PJRT_DeviceDescription_DebugString_Args;
typedef struct { size_t struct_size; PJRT_DeviceDescription* device_description; /*out*/ const char* to_string; size_t to_string_size; } PJRT_DeviceDescription_ToString_Args;
typedef struct { size_t struct_size; PJRT_DeviceDescription* device_description; /*out*/ const PJRT_NamedValue* attributes; size_t num_attributes; } PJRT_DeviceDescription_Attributes_Args;

// Memory
typedef struct { size_t struct_size; PJRT_Memory* memory; /*out*/ int id; } PJRT_Memory_Id_Args;
typedef struct { size_t struct_size; PJRT_Memory* memory; /*out*/ const char* kind; size_t kind_size; } PJRT_Memory_Kind_Args;
typedef struct { size_t struct_size; PJRT_Memory* memory; /*out*/ const char* debug_string; size_t debug_string_size; } PJRT_Memory_DebugString_Args;
typedef struct { size_t struct_size; PJRT_Memory* memory; /*out*/ const char* to_string; size_t to_string_size; } PJRT_Memory_ToString_Args;
typedef struct { size_t struct_size; PJRT_Memory* memory; /*out*/ PJRT_Device* const* devices; size_t num_devices; } PJRT_Memory_AddressableByDevices_Args;

// Buffer
typedef struct { size_t struct_size; PJRT_Buffer* buffer; } PJRT_Buffer_Destroy_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ PJRT_Buffer_Type type; } PJRT_Buffer_ElementType_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ const int64_t* dims; size_t num_dims; } PJRT_Buffer_Dimensions_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ const int64_t* dims; size_t num_dims; } PJRT_Buffer_UnpaddedDimensions_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ size_t on_device_size_in_bytes; } PJRT_Buffer_OnDeviceSizeInBytes_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ PJRT_Device* device; } PJRT_Buffer_Device_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ PJRT_Memory* memory; } PJRT_Buffer_Memory_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ bool is_deleted; } PJRT_Buffer_IsDeleted_Args;
typedef struct { size_t struct_size; PJRT_Buffer* src; void* dst; size_t dst_size; /*out*/ PJRT_Event* event; } PJRT_Buffer_ToHostBuffer_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ bool is_on_cpu; } PJRT_Buffer_IsOnCpu_Args;
typedef struct { size_t struct_size; PJRT_Buffer* buffer; /*out*/ PJRT_Event* event; } PJRT_Buffer_ReadyEvent_Args;

// Executable
typedef struct { size_t struct_size; PJRT_Executable* executable; } PJRT_Executable_Destroy_Args;
typedef struct { size_t struct_size; PJRT_Executable* executable; /*out*/ const char* name; size_t name_size; } PJRT_Executable_Name_Args;
typedef struct { size_t struct_size; PJRT_Executable* executable; /*out*/ size_t num_outputs; } PJRT_Executable_NumOutputs_Args;
typedef struct { size_t struct_size; PJRT_Executable* executable; /*out*/ int64_t size_in_bytes; } PJRT_Executable_SizeOfGeneratedCodeInBytes_Args;
typedef struct { size_t struct_size; PJRT_LoadedExecutable* loaded_executable; /*out*/ const char* fingerprint; size_t fingerprint_size; } PJRT_Executable_Fingerprint_Args;

// LoadedExecutable
typedef struct { size_t struct_size; PJRT_LoadedExecutable* executable; } PJRT_LoadedExecutable_Destroy_Args;
typedef struct { size_t struct_size; PJRT_LoadedExecutable* executable; /*out*/ PJRT_Executable* unloaded_executable; } PJRT_LoadedExecutable_GetExecutable_Args;
typedef struct { size_t struct_size; PJRT_LoadedExecutable* executable; /*out*/ PJRT_Device* const* addressable_devices; size_t num_addressable_devices; } PJRT_LoadedExecutable_AddressableDevices_Args;
typedef struct {
  size_t struct_size;
  PJRT_LoadedExecutable* executable;
  size_t num_devices; size_t num_args;
  PJRT_Buffer* const* const* argument_lists;
  /*out*/ PJRT_Buffer** const* output_lists;
  PJRT_Event** device_complete_events;
  bool execute_device;
} PJRT_LoadedExecutable_Execute_Args;

// Event
typedef struct { size_t struct_size; PJRT_Event* event; } PJRT_Event_Destroy_Args;
typedef struct { size_t struct_size; PJRT_Event* event; /*out*/ bool is_ready; } PJRT_Event_IsReady_Args;
typedef struct { size_t struct_size; PJRT_Event* event; /*out*/ PJRT_Error* error; } PJRT_Event_Error_Args;
typedef struct { size_t struct_size; PJRT_Event* event; } PJRT_Event_Await_Args;
typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error* error, void* user_arg);
typedef struct { size_t struct_size; PJRT_Event* event; PJRT_Event_OnReadyCallback callback; void* user_arg; } PJRT_Event_OnReady_Args;

// ============================================================
// PJRT_Api function pointer table
//
// WARNING: This is a SIMPLIFIED layout for Phase 1 development.
// The real PJRT_Api from XLA has ~100+ fields in a specific order.
// Before testing against real jaxlib, you MUST replace this header
// with the exact one from your XLA version. Use scripts/fetch_pjrt_header.sh.
// ============================================================

typedef struct PJRT_Api {
  size_t struct_size;
  void* extension_start;
  int pjrt_api_version_major;
  int pjrt_api_version_minor;

  PJRT_Error* (*PJRT_Error_Destroy)(PJRT_Error_Destroy_Args*);
  void (*PJRT_Error_Message)(PJRT_Error_Message_Args*);
  PJRT_Error* (*PJRT_Error_GetCode)(PJRT_Error_GetCode_Args*);

  PJRT_Error* (*PJRT_Plugin_Initialize)(PJRT_Plugin_Initialize_Args*);
  PJRT_Error* (*PJRT_Plugin_Attributes)(PJRT_Plugin_Attributes_Args*);

  PJRT_Error* (*PJRT_Client_Create)(PJRT_Client_Create_Args*);
  PJRT_Error* (*PJRT_Client_Destroy)(PJRT_Client_Destroy_Args*);
  PJRT_Error* (*PJRT_Client_PlatformName)(PJRT_Client_PlatformName_Args*);
  PJRT_Error* (*PJRT_Client_ProcessIndex)(PJRT_Client_ProcessIndex_Args*);
  PJRT_Error* (*PJRT_Client_PlatformVersion)(PJRT_Client_PlatformVersion_Args*);
  PJRT_Error* (*PJRT_Client_Devices)(PJRT_Client_Devices_Args*);
  PJRT_Error* (*PJRT_Client_AddressableDevices)(PJRT_Client_AddressableDevices_Args*);
  PJRT_Error* (*PJRT_Client_Compile)(PJRT_Client_Compile_Args*);
  PJRT_Error* (*PJRT_Client_BufferFromHostBuffer)(PJRT_Client_BufferFromHostBuffer_Args*);

  PJRT_Error* (*PJRT_DeviceDescription_Id)(PJRT_DeviceDescription_Id_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_ProcessIndex)(PJRT_DeviceDescription_ProcessIndex_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_Kind)(PJRT_DeviceDescription_Kind_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_DebugString)(PJRT_DeviceDescription_DebugString_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_ToString)(PJRT_DeviceDescription_ToString_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_Attributes)(PJRT_DeviceDescription_Attributes_Args*);

  PJRT_Error* (*PJRT_Device_GetDescription)(PJRT_Device_GetDescription_Args*);
  PJRT_Error* (*PJRT_Device_IsAddressable)(PJRT_Device_IsAddressable_Args*);
  PJRT_Error* (*PJRT_Device_LocalHardwareId)(PJRT_Device_LocalHardwareId_Args*);
  PJRT_Error* (*PJRT_Device_MemorySpaces)(PJRT_Device_MemorySpaces_Args*);
  PJRT_Error* (*PJRT_Device_DefaultMemory)(PJRT_Device_DefaultMemory_Args*);

  PJRT_Error* (*PJRT_Memory_Id)(PJRT_Memory_Id_Args*);
  PJRT_Error* (*PJRT_Memory_Kind)(PJRT_Memory_Kind_Args*);
  PJRT_Error* (*PJRT_Memory_DebugString)(PJRT_Memory_DebugString_Args*);
  PJRT_Error* (*PJRT_Memory_ToString)(PJRT_Memory_ToString_Args*);
  PJRT_Error* (*PJRT_Memory_AddressableByDevices)(PJRT_Memory_AddressableByDevices_Args*);

  PJRT_Error* (*PJRT_Buffer_Destroy)(PJRT_Buffer_Destroy_Args*);
  PJRT_Error* (*PJRT_Buffer_ElementType)(PJRT_Buffer_ElementType_Args*);
  PJRT_Error* (*PJRT_Buffer_Dimensions)(PJRT_Buffer_Dimensions_Args*);
  PJRT_Error* (*PJRT_Buffer_UnpaddedDimensions)(PJRT_Buffer_UnpaddedDimensions_Args*);
  PJRT_Error* (*PJRT_Buffer_OnDeviceSizeInBytes)(PJRT_Buffer_OnDeviceSizeInBytes_Args*);
  PJRT_Error* (*PJRT_Buffer_Device)(PJRT_Buffer_Device_Args*);
  PJRT_Error* (*PJRT_Buffer_Memory)(PJRT_Buffer_Memory_Args*);
  PJRT_Error* (*PJRT_Buffer_IsDeleted)(PJRT_Buffer_IsDeleted_Args*);
  PJRT_Error* (*PJRT_Buffer_ToHostBuffer)(PJRT_Buffer_ToHostBuffer_Args*);
  PJRT_Error* (*PJRT_Buffer_IsOnCpu)(PJRT_Buffer_IsOnCpu_Args*);
  PJRT_Error* (*PJRT_Buffer_ReadyEvent)(PJRT_Buffer_ReadyEvent_Args*);

  PJRT_Error* (*PJRT_Executable_Destroy)(PJRT_Executable_Destroy_Args*);
  PJRT_Error* (*PJRT_Executable_Name)(PJRT_Executable_Name_Args*);
  PJRT_Error* (*PJRT_Executable_NumOutputs)(PJRT_Executable_NumOutputs_Args*);
  PJRT_Error* (*PJRT_Executable_SizeOfGeneratedCodeInBytes)(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args*);
  PJRT_Error* (*PJRT_Executable_Fingerprint)(PJRT_Executable_Fingerprint_Args*);

  PJRT_Error* (*PJRT_LoadedExecutable_Destroy)(PJRT_LoadedExecutable_Destroy_Args*);
  PJRT_Error* (*PJRT_LoadedExecutable_GetExecutable)(PJRT_LoadedExecutable_GetExecutable_Args*);
  PJRT_Error* (*PJRT_LoadedExecutable_AddressableDevices)(PJRT_LoadedExecutable_AddressableDevices_Args*);
  PJRT_Error* (*PJRT_LoadedExecutable_Execute)(PJRT_LoadedExecutable_Execute_Args*);

  PJRT_Error* (*PJRT_Event_Destroy)(PJRT_Event_Destroy_Args*);
  PJRT_Error* (*PJRT_Event_IsReady)(PJRT_Event_IsReady_Args*);
  PJRT_Error* (*PJRT_Event_Error)(PJRT_Event_Error_Args*);
  PJRT_Error* (*PJRT_Event_Await)(PJRT_Event_Await_Args*);
  PJRT_Error* (*PJRT_Event_OnReady)(PJRT_Event_OnReady_Args*);
} PJRT_Api;

// Every PJRT plugin must export this symbol.
const PJRT_Api* GetPjrtApi();

#ifdef __cplusplus
}
#endif

#endif  // PJRT_C_API_H_
