// TtBuffer: wraps a ttnn tensor for PJRT buffer management.
//
// Phase 1: Stub only. Real implementation in Phase 2 will handle
// host-to-device (BufferFromHostBuffer) and device-to-host (ToHostBuffer).

#ifndef TT_PJRT_BUFFER_H_
#define TT_PJRT_BUFFER_H_

#include "pjrt_c_api.h"
#include <vector>

struct PJRT_Buffer {
  // ttnn tensor handle (void* to avoid header dependency)
  void* tt_tensor;

  // Metadata cached at creation time
  PJRT_Buffer_Type element_type;
  std::vector<int64_t> dims;
  size_t size_bytes;

  // Back-pointers
  PJRT_Device* device;
  PJRT_Memory* memory;

  bool deleted;
};

#endif  // TT_PJRT_BUFFER_H_
