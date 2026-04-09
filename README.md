# CellForge Engine (Rust + WGPU)

High-performance CellForge runtime with a trainable weight pipeline:
- real-time GPU simulation via WGPU compute shader
- interactive destruction with mouse and self-repair behavior
- offline training script (PyTorch) with binary export to Rust runtime

The runtime package is named `cellforge_engine`, while the native C ABI and shared-library target stay on the legacy `nca_engine` namespace for compatibility.

## Why this architecture is the best for speed and stability

1. Train offline, infer online:
- training uses mature autodiff stack (PyTorch)
- runtime stays lean and fast in Rust/WGPU

2. Stable CellForge update rule:
- residual update with bounded delta
- stochastic fire-rate gating
- alive mask and decay for dead zones

3. Throughput-focused GPU path:
- ping-pong storage buffers
- no CPU readback in frame loop
- configurable compute steps per frame

## Project layout

- `src/main.rs` - runtime, window loop, WGPU pipelines, interaction
- `src/compute.wgsl` - CellForge update rule on GPU
- `src/render.wgsl` - fullscreen visualization
- `src/model.rs` - trainable weight format loader/saver
- `tools/train_nca.py` - offline trainer and weight exporter

## Runtime usage

Run with `CELLFORGE_WEIGHTS` (legacy alias: `NCA_WEIGHTS`) or fallback seeded weights at `assets/nca_weights.bin`:

```bash
cargo run --release
```

Load a specific weights file:

```bash
cargo run --release -- --weights assets/nca_weights.bin
```

Tune simulation substeps:

```bash
cargo run --release -- --steps 6
```

Export seeded baseline weights and exit:

```bash
cargo run --release -- --export-default-weights assets/nca_weights.bin
```

## Training usage

Install trainer dependencies:

```bash
python -m pip install -r tools/requirements-train.txt
```

Train and export:

```bash
python tools/train_nca.py --epochs 1200 --output assets/nca_weights.bin
```

Then run the Rust runtime with the trained weights:

```bash
cargo run --release -- --weights assets/nca_weights.bin
```

## C-API SDK (FFI)

Rust library outputs are enabled in `Cargo.toml` as:

- `cdylib`
- `staticlib`
- `rlib`

Build the SDK artifacts:

```bash
cargo build --release --lib
```

Generated artifacts (Windows):

- `target/release/nca_engine.dll`
- `target/release/nca_engine.lib`

These artifact names stay unchanged so existing C/C++ and UE integrations continue to load the same binary names.

Primary C header:

- `ffi/include/nca_engine.h`

Core lifecycle API:

- `nca_engine_create`
- `nca_engine_update`
- `nca_engine_render`
- `nca_engine_destroy`

Hot-reload API:

- `nca_engine_start_hot_reload`
- `nca_engine_stop_hot_reload`

Interactive host inputs:

- `nca_engine_inject_damage`
- `nca_engine_inject_growth`

## Memory Ownership Model (Rust <-> C/C++)

The FFI uses an opaque pointer model:

1. `nca_engine_create` allocates Rust state and returns `NcaEngine*`.
2. C/C++ treats `NcaEngine*` as opaque and never dereferences it.
3. `nca_engine_destroy` is the only valid deallocation path.
4. Internal mutable state is synchronized with `Mutex` for thread-safe host calls.
5. Error text stays owned by Rust; host copies it via `nca_engine_copy_last_error`.

The function names stay under the `nca_engine_*` compatibility namespace by design.

This model avoids cross-language allocator mismatch and prevents UAF/double-free when API contract is respected.

## Unreal Engine 5 Sample ThirdParty Glue

Sample files are included in:

- `unreal/NcaEngineThirdParty/NcaEngineThirdParty.Build.cs`
- `unreal/NcaEngineThirdParty/include/nca_engine.h`
- `unreal/NcaEngineThirdParty/Public/NcaEngineBridge.h`

See integration notes in:

- `unreal/NcaEngineThirdParty/README.md`

## Product Execution Docs

To run an SDK-first commercialization track, use:

- `docs/SDK_FIRST_SHOWCASE_PILOT_MASTERPLAN.md`
- `docs/SPRINT_BACKLOG_6_WEEKS.md`
- `docs/go-to-market/PILOT_PLAYBOOK.md`
