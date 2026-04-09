#ifndef NCA_ENGINE_H
#define NCA_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(NCA_ENGINE_DLL)
    #define NCA_API __declspec(dllexport)
  #else
    #define NCA_API __declspec(dllimport)
  #endif
#else
  #define NCA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NCA_WINDOW_PLATFORM_NONE 0u
#define NCA_WINDOW_PLATFORM_WINDOWS_HWND 1u

typedef enum NcaStatus {
  NCA_STATUS_OK = 0,
  NCA_STATUS_NULL_POINTER = 1,
  NCA_STATUS_INVALID_ARGUMENT = 2,
  NCA_STATUS_UNSUPPORTED = 3,
  NCA_STATUS_RUNTIME_ERROR = 4,
} NcaStatus;

typedef struct NcaEngineOpaque NcaEngine;

typedef struct NcaHostWindowHandle {
  uint32_t platform;
  uint64_t window_handle;
  uint64_t display_handle;
  uint32_t width;
  uint32_t height;
} NcaHostWindowHandle;

typedef struct NcaEngineConfig {
  uint32_t grid_width;
  uint32_t grid_height;
  uint32_t surface_width;
  uint32_t surface_height;
  uint32_t steps_per_frame;
  uint32_t prefer_mailbox;
  uint32_t create_internal_window;
  NcaHostWindowHandle host_window;
  const char* weights_path;
} NcaEngineConfig;

typedef struct NcaBrushEvent {
  float x;
  float y;
  float radius;
  float strength;
  uint32_t duration_frames;
} NcaBrushEvent;

NCA_API NcaStatus nca_engine_create(const NcaEngineConfig* config, NcaEngine** out_engine);
NCA_API void nca_engine_destroy(NcaEngine* engine);

NCA_API NcaStatus nca_engine_update(NcaEngine* engine, float delta_seconds);
NCA_API NcaStatus nca_engine_render(NcaEngine* engine);
NCA_API NcaStatus nca_engine_resize_surface(NcaEngine* engine, uint32_t width, uint32_t height);
NCA_API NcaStatus nca_engine_set_steps_per_frame(NcaEngine* engine, uint32_t steps_per_frame);

NCA_API NcaStatus nca_engine_load_weights(NcaEngine* engine, const char* weights_path);
NCA_API NcaStatus nca_engine_load_weights_from_memory(
  NcaEngine* engine,
  const uint8_t* data,
  size_t data_len
);

NCA_API NcaStatus nca_engine_inject_damage(NcaEngine* engine, const NcaBrushEvent* brush_event);
NCA_API NcaStatus nca_engine_inject_growth(NcaEngine* engine, const NcaBrushEvent* brush_event);

NCA_API NcaStatus nca_engine_start_hot_reload(
  NcaEngine* engine,
  const char* weights_path,
  uint32_t debounce_ms
);
NCA_API NcaStatus nca_engine_stop_hot_reload(NcaEngine* engine);

/*
Copies the last error message into dst as a null-terminated string.
Returns required bytes including the null terminator.
If dst is NULL or dst_len is 0, no copy occurs and only the required length is returned.
*/
NCA_API size_t nca_engine_copy_last_error(NcaEngine* engine, char* dst, size_t dst_len);

NCA_API uint32_t nca_engine_version_major(void);
NCA_API uint32_t nca_engine_version_minor(void);
NCA_API uint32_t nca_engine_version_patch(void);

#ifdef __cplusplus
}
#endif

#endif /* NCA_ENGINE_H */
