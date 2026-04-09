# CellForge C-API SDK Notes

CellForge Engine keeps the legacy `nca_engine_*` ABI surface so existing integrations continue to work without source changes.

## Rust build targets

`Cargo.toml` exposes:

- `cdylib` for dynamic linking from C/C++
- `staticlib` for static linking in native toolchains
- `rlib` for Rust-internal reuse

## Public C-API header

- `ffi/include/nca_engine.h`

## Core ownership model

- Rust owns the full engine state in an opaque object (`NcaEngineOpaque`).
- C/C++ receives only `NcaEngine*` and never dereferences internals.
- `nca_engine_create` allocates with `Box::into_raw`.
- `nca_engine_destroy` must be called exactly once to reclaim memory.

## Thread-safety model

- Shared mutable state is guarded by `Mutex` in Rust.
- FFI calls can come from host threads without violating Rust aliasing rules.
- Hot-reload worker runs in background and sends decoded weights through channels.
- Weight upload to GPU is applied on the update thread during `nca_engine_update`.

## Hot-reload behavior

- `nca_engine_start_hot_reload` watches file changes via `notify`.
- Debounce interval reduces repeated recompilation/reload spikes.
- Reload does not stop simulation; new weights are swapped into GPU buffers live.

## Zero-copy considerations

- The API provides `nca_engine_load_weights_from_memory` to avoid filesystem I/O.
- GPU upload still performs one transfer from host memory to VRAM, which is expected.
