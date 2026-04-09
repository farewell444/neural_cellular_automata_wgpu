# CellForge Unreal ThirdParty Integration (Sample)

This folder contains a minimal UE5 ThirdParty bridge for CellForge Engine.

The sample still links the legacy `nca_engine` library/header names so it stays compatible with existing integrations.

## Expected layout

Place compiled Rust artifacts here for Win64:

- `unreal/NcaEngineThirdParty/lib/Win64/nca_engine.lib`
- `unreal/NcaEngineThirdParty/lib/Win64/nca_engine.dll`

C header for C-API is in:

- `unreal/NcaEngineThirdParty/include/nca_engine.h`

Optional UE C++ helper wrapper:

- `unreal/NcaEngineThirdParty/Public/NcaEngineBridge.h`

## Build.cs

The sample module file:

- `unreal/NcaEngineThirdParty/NcaEngineThirdParty.Build.cs`

adds include path, links `nca_engine.lib`, and delay-loads `nca_engine.dll`.

## Host window wiring

For UE integration, pass:

- `platform = NCA_WINDOW_PLATFORM_WINDOWS_HWND`
- `window_handle = (uint64_t)HWND`
- `display_handle = (uint64_t)HINSTANCE` (or 0)

through `NcaEngineConfig.host_window`.
