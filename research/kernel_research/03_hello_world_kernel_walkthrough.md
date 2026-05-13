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

Line by line:

- `get_arg_val<uint32_t>(k)` pulls runtime arg slot `k`. Order must match `SetRuntimeArgs` on the host.
- `TensorAccessorArgs<0>()` — *compile-time* template indexed into the `compile_args` blob the host packed. `<0>` means "first TensorAccessor metadata." The second one uses `in0_args.next_compile_time_args_offset()` to skip past the first.
- `TensorAccessor(args, base_addr)` is a stateless functor. Calling it on tile-id `i` returns the NoC address for tile `i` taking bank interleaving into account.
- `noc_async_read_tile(i, in0, l1_buffer_addr)` issues a NoC read of tile `i` from `in0`'s buffer into the L1 slot at `l1_buffer_addr`. Asynchronous — returns immediately.
- `noc_async_read_barrier()` blocks the RISC-V until all outstanding reads on this core's read channel complete. Required before we use the data.
- `noc_async_write_tile` mirrors the read but goes L1 → DRAM. The same L1 slot is reused next iteration because the barriers guarantee no in-flight traffic.

Notice what is **absent**: no circular buffer (`cb_*`), no `tile_regs_acquire`, no math engine, no `pack_tile`. Pure data movement — exactly the floor we want before adding compute.

## 4. Where data flows

```
host input_vec  --(EnqueueWriteMeshBuffer)-->  input_dram_buffer (DRAM, 50 tiles, interleaved across banks)
                                                      |
                                                      v   noc_async_read_tile(i)
                                                  l1_buffer (L1 on core {0,0}, 1 tile = 2 KB)
                                                      |
                                                      v   noc_async_write_tile(i)
                                              output_dram_buffer  --(EnqueueReadMeshBuffer)--> result_vec
```

- **DRAM addresses** are derived inside the kernel by `TensorAccessor` from (compile-time bank layout + runtime `address()` of the buffer + tile id).
- **Runtime args** are a flat `uint32_t` array — pure values, not references. Set once per `SetRuntimeArgs`, baked into the program until overwritten.
- **CBs are not configured** here — they only show up once compute (math) joins the pipeline.

## 5. What changes for our scatter kernel

Spec recap: cache `[1, 4, 256, 256]` bf16 (resident in DRAM), src `[1, 4, 1, 256]` bf16, write src into `cache[:, :, cur_pos, :]`. No reduction, no math.

| Pattern in loopback | What we keep | What we change for scatter |
|---|---|---|
| One reader on `{0,0}`, BRISC, no compute | Keep — scatter is also pure data movement | Same single-core start; parallelize across heads later. |
| `num_tiles` linear loop | Loop structure | Loop over `head=0..3`; each iter copies 256 bf16 = 8 tiles of src into one row of the cache tile-grid. |
| Source/dest are independent buffers | Two `TensorAccessor`s | **Same buffer for read and write?** Cache is read-modify-write at row `cur_pos`. Speculation: simplest path is to *only write* the new row — no read needed if the rest of the cache is untouched. So scatter is one `TensorAccessor` over `cache`, plus one over `src`. |
| Static tile index `i` | Tile-id arithmetic | Index becomes `tile_id = head * tiles_per_head + (cur_pos / 32) * tiles_per_row + col`. This is the new logic we must derive carefully — `cur_pos` is **mid-tile** (rows are packed 32-per-tile), so we cannot just plop a whole tile. Two options: (a) widen to write a full tile by reading the existing tile, blending the new row in, writing back; (b) use `noc_async_write` with sub-tile byte addressing. (Speculation — needs experiment.) |
| `cur_pos` constant per launch | Runtime arg | Pass `cur_pos` as `get_arg_val<uint32_t>(N)`. Avoid baking it into compile args so we don't recompile per step. (Echoes the "Trace Capture" lesson from MEMORY: scalars must be device-side, not Python-side.) |
| Tile layout 32x32 of bf16 | Same | Cache is `[..., 256, 256]` → tile-grid `8 x 8` per (batch,head). `src` is 1 row of 256 elements → straddles 8 tiles of width but only 1 of 32 rows — sub-tile write territory. |
| Single barrier per op | Keep | Same. Read-modify-write needs read barrier *before* the modify, write barrier *after*. |
| No CB | Keep | Still no CB needed — pure NoC traffic. |
| `TensorAccessorArgs` from host | Keep | Host packs accessors for `cache` and `src` at compile time; runtime args carry base addresses + `cur_pos` + (batch, head sizes if not compile-time). |

**Key new design question (label: speculation):** does `noc_async_write_tile` allow writing only part of a tile, or must we round-trip a full tile through L1 to preserve the 31 other rows? If the latter, our kernel becomes: per-head, per-column-tile: read tile from cache, overwrite the 1 row corresponding to `cur_pos % 32`, write tile back. That is structurally `loopback` with a one-row blend in the middle — a very small delta to verify against.

**Reference for next step:** when we are ready to write the kernel, copy `loopback/` wholesale and replace the inner loop body. The host plumbing is essentially reusable verbatim.
