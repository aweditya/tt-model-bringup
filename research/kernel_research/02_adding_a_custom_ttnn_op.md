# Adding a custom TTNN op: full pipeline (`.cpp` -> `ttnn.foo()`)

Source of truth: `tt_docs_corpus/.../ttnn/ttnn/adding_new_ttnn_operation.md` plus the live
example in `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/examples/example/` and the
in-tree `kv_cache` and `experimental/paged_cache` ops, which are the closest analogues to
the planned `in_place_scatter_kv`.

This is a recipe for adding `ttnn_local.in_place_scatter_kv(cache, src_row, position)`
that writes a row into a KV cache in place. The TT-NN tree does not have a
`ttnn_local.*` namespace today; we either (a) live under `ttnn.experimental.*` or
`ttnn.kv_cache.*`, or (b) ship a separate Python module that wraps a stand-alone op
library. See section 3 on the build system for which option is realistic.

## 1. Directory layout

A device operation in TT-NN has a strict layout. The example op
(`ttnn/cpp/ttnn/operations/examples/example/`) shows the canonical shape:

```
ttnn/cpp/ttnn/operations/<category>/<op_name>/
  <op_name>.hpp                # public C++ entry (the ttnn::op symbol)
  <op_name>.cpp                # composite glue: builds attrs/args, calls launch
  <op_name>_nanobind.hpp       # forward-declares bind_<op>(nb::module_&)
  <op_name>_nanobind.cpp       # nanobind binding: ttnn::bind_function<"name", "ttnn.cat.">
  device/
    <op_name>_device_operation.hpp   # DeviceOperationConcept struct
    <op_name>_device_operation.cpp   # validate / compute_output_specs / create_output_tensors
    single_core_program_factory.cpp  # >=1 program factory; multi_core_program_factory.cpp etc.
    kernels/
      dataflow/reader_*.cpp
      dataflow/writer_*.cpp
      compute/*.cpp                  # only if compute kernels are needed
  CMakeLists.txt               # target_sources + install rules for this op-lib
  sources.cmake                # list of .cpp files in this op
```

For `in_place_scatter_kv` the closest pattern is `ttnn/cpp/ttnn/operations/kv_cache/`
(top-level `update_cache`, `fill_cache`, `zero_cache_range`) and the experimental
`paged_cache/` variant. Both already mutate the cache in place and return the cache
tensor handle, which is exactly the semantic we want.

The minimum required pieces are: the `*_device_operation.{hpp,cpp}` struct, at least one
`*_program_factory.cpp`, the `kernels/dataflow/{reader,writer}_*.cpp` kernels, the
top-level `op.{hpp,cpp}` entry, and the nanobind file.

## 2. Registration into `ttnn.*`

Two layers of registration. From `adding_new_ttnn_operation.md`:

- C++ side: either expose a plain function in `namespace ttnn { Tensor my_op(...); }` (the
  `kv_cache` and `paged_cache` style) or use the templated decorator
  `ttnn::register_operation<"ttnn::my_op", MyOpStruct>()` (the composite example style).
  The `register_operation` decorator auto-creates the Python attribute.

- Python side: `ttnn::bind_function<"name", "ttnn.category.">(mod, doc, &ttnn::fn, args...)`
  inside an `_nanobind.cpp`. The second template argument is the Python module path. The
  doc states: "If the operation is called `ttnn::add` in C++, then the python binding will
  be `ttnn.add`."

Concrete patterns observed:

- `kv_cache/kv_cache_nanobind.cpp` binds `update_cache` directly into `ttnn.*` and
  `update_cache_for_token_` into `ttnn.kv_cache.*` via the `"ttnn.kv_cache."` prefix.
- `experimental/paged_cache/paged_cache_nanobind.cpp` binds with prefix
  `"ttnn.experimental."`, putting the op at `ttnn.experimental.paged_update_cache`.

To create a `ttnn_local` namespace specifically, we would either alias inside our Python
wrapper (`ttnn_local = SimpleNamespace(in_place_scatter_kv=ttnn.experimental.in_place_scatter_kv)`)
or do it the upstream way and submit a PR landing under `ttnn.experimental.kv_*` or
`ttnn.kv_cache.*`.

## 3. Build / install: integrated vs side-loaded

Each op-category lives in its own static library wired into the ttnn build. From
`ttnn/cpp/ttnn/operations/examples/CMakeLists.txt`:

```
add_library(ttnn_op_examples ${LIB_TYPE})
add_library(TTNN::Ops::Examples ALIAS ttnn_op_examples)
target_link_libraries(ttnn_op_examples PRIVATE TT::Metalium TTNN::Core)
install(TARGETS ttnn_op_examples LIBRARY COMPONENT tar)
```

with sources listed in a sibling `sources.cmake`. To add a new op you:
1. create the op directory under `ttnn/cpp/ttnn/operations/<category>/`,
2. add a `CMakeLists.txt` + `sources.cmake` modelled on the examples one,
3. wire the new lib into the parent `ttnn/cpp/ttnn/operations/CMakeLists.txt`,
4. call the new `bind_*_operation(mod)` from the category-level `*_nanobind.cpp`,
5. rebuild ttnn from source: `./build_metal.sh` then reinstall the wheel.

So the path of least resistance for an in-tree op is **rebuild ttnn**. There is no
documented sideload story in `adding_new_ttnn_operation.md` — the doc assumes you are a
TT-NN contributor and patching the source tree. **Needs investigation**: whether you can
build a standalone shared library that links `TT::Metalium` + `TTNN::Core` from an
externally installed ttnn dev-package and `dlopen` it from Python. The CMake targets are
installed (`install(TARGETS ttnn_op_examples ...)`) so the headers and libraries are on
disk under the install prefix, but the docs do not describe an "external op SDK".

For our project the realistic move is to fork or vendor tt-metal under
`experiments/.refs/tt-metal/`, add the op there, and rebuild ttnn. That fits the
"correctness first" principle and lets us use Tracy/profiling without surprises.

## 4. Unit testing infrastructure

Tests are plain pytest files under `tests/ttnn/unit_tests/operations/`. Examples:
- `tests/ttnn/unit_tests/operations/test_zero_cache_range.py` for `ttnn.kv_cache.zero_cache_range`
- `tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py`

Pattern: a `device` pytest fixture supplies the open device, the test builds torch
references, calls `ttnn.from_torch -> ttnn.my_op -> ttnn.to_torch`, and asserts via
`pcc`/`allclose`. Optionally attach a Python `golden_function` with
`ttnn.attach_golden_function(ttnn.example, golden_function=...)` so the framework can
diff against torch automatically when `TTNN_VALIDATE_OUTPUT=1` (see step 2 of the Python
section of the docs).

A documentation/CI side-channel exists at
`tests/ttnn/docs_examples/examples_mapping.py` — a `FUNCTION_TO_EXAMPLES_MAPPING_DICT`
maps each op to an example pytest that doubles as the rendered doc snippet.

For our project we can also write the test as a stand-alone script under
`pjrt_plugin/tests/` and run it via `ssh qb1`, independent of tt-metal's CI.

## 5. Experimental vs production namespace

There is no formal stability gate; the difference is purely the binding prefix and
header namespace:

| Tier         | C++ namespace          | Python path                        | Stability signal                                   |
| ------------ | ---------------------- | ---------------------------------- | -------------------------------------------------- |
| Production   | `ttnn::`               | `ttnn.foo` / `ttnn.kv_cache.foo`   | Stable signature, full docs, on the `api/` index   |
| Experimental | `ttnn::experimental::` | `ttnn.experimental.foo`            | Allowed to change; lives under `operations/experimental/` |

Mechanically:
- bind prefix in nanobind: `ttnn::bind_function<"paged_update_cache", "ttnn.experimental.">`
- C++ namespace: `namespace ttnn::experimental { Tensor paged_update_cache(...); }`
- directory: `ttnn/cpp/ttnn/operations/experimental/<op>/`

For `in_place_scatter_kv` we should **start under `ttnn.experimental`** until the
signature is locked. Once the op is validated and the PJRT lowering depends on it, we
can promote by moving the directory out of `experimental/`, changing the nanobind
prefix, and updating the namespace.

## Recipe summary (checklist for `in_place_scatter_kv`)

1. Create `ttnn/cpp/ttnn/operations/experimental/in_place_scatter_kv/` with the file
   layout from section 1 (mirror `paged_cache` for in-place semantics).
2. Define `struct InPlaceScatterKvOperation` satisfying `DeviceOperationConcept`:
   `operation_attributes_t { uint32_t position; }`,
   `tensor_args_t { Tensor& cache; const Tensor& src_row; }`,
   return type = `Tensor` (the same cache handle).
3. One `*_program_factory.cpp` with a writer-only kernel under
   `device/kernels/dataflow/writer_in_place_scatter_kv.cpp`. Reader can be reused from
   the kv_cache op if shapes match.
4. Top-level `in_place_scatter_kv.cpp` exposing `namespace ttnn::experimental { Tensor in_place_scatter_kv(...); }`.
5. `in_place_scatter_kv_nanobind.cpp` using `bind_function<"in_place_scatter_kv", "ttnn.experimental.">`.
6. New `CMakeLists.txt` + `sources.cmake`; register in the experimental category
   `CMakeLists.txt` and `experimental_nanobind.cpp`.
7. Rebuild ttnn (`./build_metal.sh`) and reinstall the wheel; verify
   `ttnn.experimental.in_place_scatter_kv` is callable.
8. Add `tests/ttnn/unit_tests/operations/transformers/test_in_place_scatter_kv.py`
   with a torch golden + pcc assert; optionally `attach_golden_function`.
9. Once stable, alias as `ttnn_local.in_place_scatter_kv` in our Python wrapper or
   promote into `ttnn.kv_cache.*`.

## Open questions / needs investigation

- Whether a true stand-alone shared library (no ttnn-tree fork) can register an op into
  the running `ttnn` Python module. The docs do not describe this path. The install
  rules in `examples/CMakeLists.txt` do export the lib + kernel sources, so it is
  plausible but unverified.
- Whether `attach_golden_function` works for in-place ops where the returned tensor is
  the same handle as the input cache (the framework's auto-torch-roundtrip may copy).
- Whether putting our op under `ttnn.experimental.*` interferes with PJRT trace capture
  (most existing C-prime work already uses `ttnn.experimental.*`, so probably fine).
