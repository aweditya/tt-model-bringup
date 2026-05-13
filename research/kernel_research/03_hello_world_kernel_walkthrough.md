# Hello World tt-metal Kernel: DRAM Loopback Walkthrough

**Goal:** Before writing our first Tensix kernel (scatter-into-cache), annotate the smallest end-to-end tt-metal example to nail down the host/kernel split, runtime args, and NoC data movement.

## 1. Why "loopback" is the canonical hello-world

Two candidates under `tt_metal/programming_examples/`:

- `hello_world_datamovement_kernel/`: a RISC-V core prints `"Hello, host..."`. No DRAM, L1, or NoC. Too thin.
- `loopback/`: one Tensix core copies 50 tiles DRAM → L1 → DRAM. **This one** — it exercises every component scatter needs (NoC reads/writes, `TensorAccessor`, runtime args, L1 staging) and nothing else.

Sources: `experiments/.refs/tt-metal/tt_metal/programming_examples/loopback/loopback.cpp` (host), `loopback/kernels/loopback_dram_copy.cpp` (kernel). Docs at `tt_docs_corpus/.../tt_metal/examples/dram_loopback.md`.

## 2. Host side — `loopback.cpp`

### Device + command queue

```cpp
constexpr int device_id = 0;
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
Program program = CreateProgram();
```

- `create_unit_mesh` wraps a single chip in a 1x1 mesh so the same API scales to multi-device later.
- `MeshCommandQueue` is the only host↔device channel: uploads, downloads, and program launch are all FIFO-ordered through it.
- `Program` is an empty container; kernels get attached before launch.

### Buffer sizing + allocation

```cpp
constexpr uint32_t num_tiles = 50;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT; // 32*32
constexpr uint32_t tile_size_bytes   = sizeof(bfloat16) * elements_per_tile;                    // 2048
constexpr uint32_t dram_buffer_size  = tile_size_bytes * num_tiles;                             // 102400

distributed::DeviceLocalBufferConfig dram_config{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};
distributed::DeviceLocalBufferConfig l1_config  {.page_size = tile_size_bytes, .buffer_type = BufferType::L1};
distributed::ReplicatedBufferConfig dram_buffer_config{.size = dram_buffer_size};
distributed::ReplicatedBufferConfig l1_buffer_config  {.size = tile_size_bytes};

auto l1_buffer         = distributed::MeshBuffer::create(l1_buffer_config,   l1_config,   mesh_device.get());
auto input_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
auto output_dram_buffer= distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
```

- Tiles are `32x32` because the math engine only operates on that shape. Layout is still tile-aligned here so the same code works once compute is added.
- `page_size = tile_size_bytes` is the round-robin granularity across DRAM banks. With 6 banks (Wormhole-ish), tiles 0..5 land in banks 0..5, tile 6 wraps. `TensorAccessor` hides this from the kernel.
- The L1 buffer is **one tile** — staging slot reused each iteration.
- Two-config idiom (`DeviceLocal*` + `Replicated*`): per-device properties vs. mesh distribution. On a unit mesh `Replicated` just means "allocate once."

### Upload + attach kernel

`EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, /*blocking=*/false)` returns before DMA completes; later `Finish(cq)` enforces ordering. Host must keep `input_vec` alive until then.

```cpp
constexpr CoreCoord core = {0, 0};
std::vector<uint32_t> dram_copy_compile_time_args;
TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);

KernelHandle dram_copy_kernel_id = CreateKernel(
    program,
    "loopback/kernels/loopback_dram_copy.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc       = NOC::RISCV_0_default,
        .compile_args = dram_copy_compile_time_args});
```

- `{0,0}` is the top-left Tensix; only one core used.
- `TensorAccessorArgs` packs *compile-time* metadata (DRAM vs L1? interleaved? sharded? page size?) into a `uint32_t` blob. The kernel reconstructs `TensorAccessor` from this.
- Tensix has 5 RISC-Vs (BRISC, NCRISC, 3x TRISC). `RISCV_0` is BRISC — the canonical reader.

### Runtime args + launch

```cpp
const std::vector<uint32_t> runtime_args = {
    l1_buffer->address(),
    input_dram_buffer->address(),
    output_dram_buffer->address(),
    num_tiles};
SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range(mesh_device->shape());
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
distributed::Finish(cq);
```

- Runtime args are 4 `uint32_t`s. **DRAM "addresses" are integers, not pointers** — the RISC-V can't dereference DRAM; it issues NoC requests using these as base offsets.
- `address()` is the per-buffer base; `TensorAccessor` combines base + tile-id + page-size into the bank-local NoC address.
- Final teardown: `EnqueueReadMeshBuffer(cq, result_vec, output_dram_buffer, /*blocking=*/true)` then `mesh_device->close()`.

## 3. Kernel side — `loopback_dram_copy.cpp`

```cpp
void kernel_main() {
    std::uint32_t l1_buffer_addr       = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles            = get_arg_val<uint32_t>(3);

    constexpr auto in0_args  = TensorAccessorArgs<0>();
    const auto in0           = TensorAccessor(in0_args, dram_buffer_src_addr);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0          = TensorAccessor(out0_args, dram_buffer_dst_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        noc_async_read_tile (i, in0,  l1_buffer_addr);
        noc_async_read_barrier();
        noc_async_write_tile(i, out0, l1_buffer_addr);
        noc_async_write_barrier();
    }
}
```

- `get_arg_val<uint32_t>(k)` reads runtime arg slot `k`; order must match `SetRuntimeArgs`.
- `TensorAccessorArgs<0>()` is the *compile-time* template indexed into the `compile_args` blob the host packed. `<0>` is "first accessor's metadata"; the second uses `next_compile_time_args_offset()` to skip past it.
- `TensorAccessor(args, base_addr)` is a stateless functor. Indexed by tile id, it yields the NoC address for that tile accounting for bank interleaving.
- `noc_async_read_tile(i, in0, l1_buffer_addr)` — NoC read of tile `i` from `in0` into L1 slot. Asynchronous.
- `noc_async_read_barrier()` blocks until outstanding reads complete. Required before using the data.
- `noc_async_write_tile` mirrors it L1 → DRAM. Same L1 slot reused each iter because the barriers fence all in-flight traffic.

**Absent on purpose**: no `cb_*` (circular buffer), no `tile_regs_acquire`, no math engine, no `pack_tile`. Pure data movement.

## 4. Where data flows

```
host input_vec --EnqueueWriteMeshBuffer--> input_dram_buffer (DRAM, 50 tiles, interleaved)
                                                  | noc_async_read_tile(i)
                                                  v
                                          l1_buffer (L1 on {0,0}, 1 tile = 2 KB)
                                                  | noc_async_write_tile(i)
                                                  v
                              output_dram_buffer --EnqueueReadMeshBuffer--> result_vec
```

- **DRAM addresses** are derived in-kernel by `TensorAccessor` from (compile-time bank layout) + (runtime `address()`) + (tile id).
- **Runtime args** are a flat `uint32_t` array — values, not references. Baked into the program until overwritten by another `SetRuntimeArgs`.
- **No CBs configured** — they only appear once compute joins the pipeline.

## 5. What changes for our scatter kernel

Spec recap: cache `[1, 4, 256, 256]` bf16 (resident in DRAM), src `[1, 4, 1, 256]` bf16, write src into `cache[:, :, cur_pos, :]`. No reduction, no math.

| Loopback pattern | Scatter delta |
|---|---|
| One reader, BRISC, no compute | Same — scatter is also pure data movement. Parallelize across heads later. |
| `num_tiles` linear loop | Loop over `head=0..3`; each iter copies 256 bf16 = 8 tiles' worth into one row of the cache tile-grid for that head. |
| Two independent buffers | One `TensorAccessor` over `cache`, one over `src`. Speculation: pure-write may be possible if cache rest is untouched; otherwise it is read-modify-write of one row. |
| Static tile index `i` | Compute `tile_id = head * tiles_per_head + (cur_pos / 32) * tiles_per_row + col`. `cur_pos` is mid-tile (32 rows packed per tile), so we cannot plop a whole tile. Speculation: either (a) read tile, blend in row, write tile, or (b) use `noc_async_write` with sub-tile byte addressing — needs experiment. |
| `cur_pos` would be constant per launch | Pass as `get_arg_val<uint32_t>(N)` at runtime, never compile-time (echoes the "Trace Capture" lesson — scalars must be device-side, not Python-side). |
| Tile layout 32x32 bf16 | Cache `[..,256,256]` → 8x8 tile-grid per (batch,head). `src` is 1 row of 256 → spans 8 tile columns but only 1/32 rows. Sub-tile territory. |
| Single barrier per op | Read-modify-write needs read-barrier *before* modify, write-barrier *after*. |
| No CB | Still no CB. |
| Host packs `TensorAccessorArgs` | Same; runtime args carry base addrs + `cur_pos` + dims if not compile-time. |

**Open design question (speculation):** does `noc_async_write_tile` allow partial-tile writes, or must we round-trip full tiles through L1 to preserve the other 31 rows? If the latter, scatter is structurally `loopback` with a one-row blend in the middle — a small delta to verify.

**Next step:** copy `loopback/` wholesale and replace the inner loop body. Host plumbing is reusable verbatim.
