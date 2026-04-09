mod model;

// C-FFI SDK entry point for the Neural Cellular Automata engine.
//
// Design goals:
// - Opaque engine handles for safe cross-language ownership.
// - Thread-safe host calls through Mutex-guarded state.
// - Zero-alloc frame loop on the Rust side after initialization.
// - Hot-reload of weights without stopping simulation.

use std::ffi::{c_char, CStr};
use std::num::NonZeroIsize;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use model::{NetworkParameters, STATE_DIM};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{GetLastError, HWND};
#[cfg(windows)]
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
#[cfg(windows)]
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, DispatchMessageW, PeekMessageW, ShowWindow, TranslateMessage,
    CW_USEDEFAULT, MSG, PM_REMOVE, SW_SHOW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

const DEFAULT_GRID_WIDTH: u32 = 512;
const DEFAULT_GRID_HEIGHT: u32 = 512;
const DEFAULT_SURFACE_WIDTH: u32 = 1280;
const DEFAULT_SURFACE_HEIGHT: u32 = 720;
const DEFAULT_STEPS_PER_FRAME: u32 = 4;
const MAX_STEPS_PER_FRAME: u32 = 32;
const WORKGROUP_SIZE: u32 = 8;

pub const NCA_WINDOW_PLATFORM_NONE: u32 = 0;
pub const NCA_WINDOW_PLATFORM_WINDOWS_HWND: u32 = 1;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidArgument = 2,
    Unsupported = 3,
    RuntimeError = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NcaHostWindowHandle {
    pub platform: u32,
    pub window_handle: u64,
    pub display_handle: u64,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NcaEngineConfig {
    pub grid_width: u32,
    pub grid_height: u32,
    pub surface_width: u32,
    pub surface_height: u32,
    pub steps_per_frame: u32,
    pub prefer_mailbox: u32,
    pub create_internal_window: u32,
    pub host_window: NcaHostWindowHandle,
    pub weights_path: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NcaBrushEvent {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub strength: f32,
    pub duration_frames: u32,
}

#[repr(C)]
pub struct NcaEngineOpaque {
    // The actual Rust-owned engine state is private and never exposed to C/C++.
    handle: NcaEngineHandle,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniforms {
    grid: [u32; 4],
    params_a: [f32; 4],
    params_b: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushUniforms {
    brush: [f32; 4],
    flags: [u32; 4],
}

#[derive(Clone, Copy)]
enum BrushMode {
    Damage,
    Growth,
}

#[derive(Clone, Copy)]
struct PendingBrush {
    mode: BrushMode,
    event: NcaBrushEvent,
    frames_left: u32,
}

struct SurfaceBundle {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

#[cfg(windows)]
struct InternalWindow {
    hwnd: isize,
    hinstance: isize,
}

#[cfg(windows)]
impl InternalWindow {
    fn create(width: u32, height: u32) -> Result<Self, String> {
        let title = wide_null("Neural Cellular Automata SDK");
        let class = wide_null("STATIC");
        let hinstance = unsafe { GetModuleHandleW(ptr::null()) } as isize;

        let hwnd = unsafe {
            CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                (WS_OVERLAPPEDWINDOW | WS_VISIBLE) as u32,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width as i32,
                height as i32,
                ptr::null_mut(),
                ptr::null_mut(),
                hinstance as _,
                ptr::null(),
            )
        } as isize;

        if hwnd == 0 {
            let code = unsafe { GetLastError() };
            return Err(format!("CreateWindowExW failed with error code {code}"));
        }

        unsafe {
            ShowWindow(hwnd as HWND, SW_SHOW);
        }

        Ok(Self { hwnd, hinstance })
    }
}

#[cfg(windows)]
impl Drop for InternalWindow {
    fn drop(&mut self) {
        unsafe {
            if self.hwnd != 0 {
                let _ = DestroyWindow(self.hwnd as HWND);
            }
        }
    }
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

struct EngineCore {
    grid_width: u32,
    grid_height: u32,
    steps_per_frame: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_bundle: Option<SurfaceBundle>,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: Option<wgpu::RenderPipeline>,
    compute_bind_groups: [wgpu::BindGroup; 2],
    render_bind_groups: [wgpu::BindGroup; 2],
    _state_buffers: [wgpu::Buffer; 2],
    w1_buffer: wgpu::Buffer,
    b1_buffer: wgpu::Buffer,
    w2_buffer: wgpu::Buffer,
    b2_buffer: wgpu::Buffer,
    sim_buffer: wgpu::Buffer,
    brush_buffer: wgpu::Buffer,
    sim_uniforms: SimUniforms,
    brush_uniforms: BrushUniforms,
    active_state: usize,
    pending_brush: Option<PendingBrush>,
    #[cfg(windows)]
    internal_window: Option<InternalWindow>,
}

impl EngineCore {
    async fn new(config: ParsedConfig, network: NetworkParameters) -> Result<Self, (NcaStatus, String)> {
        let instance = wgpu::Instance::default();

        let mut surface: Option<wgpu::Surface<'static>> = None;
        let mut surface_width = config.surface_width;
        let mut surface_height = config.surface_height;

        #[cfg(windows)]
        let mut internal_window: Option<InternalWindow> = None;

        match config.surface_request {
            SurfaceRequest::None => {}
            SurfaceRequest::HostWindows {
                hwnd,
                hinstance,
                width,
                height,
            } => {
                surface = Some(create_surface_from_windows_handles(&instance, hwnd, hinstance)?);
                surface_width = width;
                surface_height = height;
            }
            SurfaceRequest::InternalWindow { width, height } => {
                #[cfg(windows)]
                {
                    let window = InternalWindow::create(width, height)
                        .map_err(|err| (NcaStatus::RuntimeError, err))?;
                    surface = Some(create_surface_from_windows_handles(
                        &instance,
                        window.hwnd,
                        window.hinstance,
                    )?);
                    surface_width = width;
                    surface_height = height;
                    internal_window = Some(window);
                }
                #[cfg(not(windows))]
                {
                    let _ = width;
                    let _ = height;
                    return Err((
                        NcaStatus::Unsupported,
                        "internal window mode is only supported on Windows for this build"
                            .to_owned(),
                    ));
                }
            }
        }

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                (
                    NcaStatus::RuntimeError,
                    "no suitable GPU adapter found".to_owned(),
                )
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("NCA FFI Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|err| {
                (
                    NcaStatus::RuntimeError,
                    format!("request_device failed: {err}"),
                )
            })?;

        let mut surface_bundle = None;
        let mut render_format = None;

        if let Some(surface) = surface {
            let caps = surface.get_capabilities(&adapter);
            let format = caps
                .formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or_else(|| caps.formats[0]);

            let present_mode = if config.prefer_mailbox
                && caps.present_modes.contains(&wgpu::PresentMode::Mailbox)
            {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            };

            let cfg = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: surface_width.max(1),
                height: surface_height.max(1),
                present_mode,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };

            surface.configure(&device, &cfg);
            render_format = Some(format);
            surface_bundle = Some(SurfaceBundle {
                surface,
                config: cfg,
            });
        }

        let initial_state = build_initial_state(config.grid_width, config.grid_height);

        let state_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI State Buffer A"),
            contents: bytemuck::cast_slice(&initial_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let state_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI State Buffer B"),
            contents: bytemuck::cast_slice(&initial_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sim_uniforms = SimUniforms {
            grid: [config.grid_width, config.grid_height, STATE_DIM as u32, 0],
            params_a: [0.18, 0.90, 0.62, 0.0],
            params_b: [0.58, 0.35, 0.08, 0.37],
        };

        let brush_uniforms = BrushUniforms {
            brush: [0.0, 0.0, 0.1, 1.0],
            flags: [0, 0, 0, 0],
        };

        let sim_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI Sim Uniform Buffer"),
            contents: bytemuck::bytes_of(&sim_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let brush_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI Brush Uniform Buffer"),
            contents: bytemuck::bytes_of(&brush_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let w1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI W1 Buffer"),
            contents: bytemuck::cast_slice(&network.w1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let b1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI B1 Buffer"),
            contents: bytemuck::cast_slice(&network.b1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let w2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI W2 Buffer"),
            contents: bytemuck::cast_slice(&network.w2),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let b2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFI B2 Buffer"),
            contents: bytemuck::cast_slice(&network.b2),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FFI Compute Bind Group Layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, false),
                    uniform_layout_entry(2),
                    uniform_layout_entry(3),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, true),
                    storage_layout_entry(6, true),
                    storage_layout_entry(7, true),
                ],
            });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FFI Render Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFI Compute Bind Group A->B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: brush_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: w1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: w2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b2_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFI Compute Bind Group B->A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: brush_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: w1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: w2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b2_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFI Render Bind Group A"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFI Render Bind Group B"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NCA Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NCA Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FFI Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FFI Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFI NCA Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "nca_step",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let render_pipeline = render_format.map(|format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("FFI NCA Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        });

        Ok(Self {
            grid_width: config.grid_width,
            grid_height: config.grid_height,
            steps_per_frame: config.steps_per_frame,
            device,
            queue,
            surface_bundle,
            compute_pipeline,
            render_pipeline,
            compute_bind_groups: [compute_bind_group_0, compute_bind_group_1],
            render_bind_groups: [render_bind_group_a, render_bind_group_b],
            _state_buffers: [state_a, state_b],
            w1_buffer,
            b1_buffer,
            w2_buffer,
            b2_buffer,
            sim_buffer,
            brush_buffer,
            sim_uniforms,
            brush_uniforms,
            active_state: 0,
            pending_brush: None,
            #[cfg(windows)]
            internal_window,
        })
    }

    fn set_steps_per_frame(&mut self, steps: u32) {
        self.steps_per_frame = steps.clamp(1, MAX_STEPS_PER_FRAME);
    }

    fn queue_brush(&mut self, mode: BrushMode, event: NcaBrushEvent) {
        let frames = event.duration_frames.max(1);
        self.pending_brush = Some(PendingBrush {
            mode,
            event,
            frames_left: frames,
        });
    }

    fn load_weights_from_path(&mut self, path: &Path) -> Result<(), (NcaStatus, String)> {
        let params = NetworkParameters::load(path)
            .map_err(|err| (NcaStatus::RuntimeError, format!("failed to load weights: {err}")))?;
        self.upload_weights(&params)
    }

    fn load_weights_from_bytes(&mut self, bytes: &[u8]) -> Result<(), (NcaStatus, String)> {
        let params = NetworkParameters::from_bytes(bytes)
            .map_err(|err| (NcaStatus::InvalidArgument, format!("invalid weight bytes: {err}")))?;
        self.upload_weights(&params)
    }

    fn upload_weights(&mut self, params: &NetworkParameters) -> Result<(), (NcaStatus, String)> {
        self.queue
            .write_buffer(&self.w1_buffer, 0, bytemuck::cast_slice(&params.w1));
        self.queue
            .write_buffer(&self.b1_buffer, 0, bytemuck::cast_slice(&params.b1));
        self.queue
            .write_buffer(&self.w2_buffer, 0, bytemuck::cast_slice(&params.w2));
        self.queue
            .write_buffer(&self.b2_buffer, 0, bytemuck::cast_slice(&params.b2));
        Ok(())
    }

    fn update(&mut self, delta_seconds: f32) -> Result<(), (NcaStatus, String)> {
        #[cfg(windows)]
        self.pump_internal_window_messages();

        self.sim_uniforms.grid[3] = self.sim_uniforms.grid[3].wrapping_add(1);
        self.sim_uniforms.params_a[3] += delta_seconds.max(0.0);

        self.refresh_brush_uniform();

        self.queue
            .write_buffer(&self.sim_buffer, 0, bytemuck::bytes_of(&self.sim_uniforms));
        self.queue
            .write_buffer(&self.brush_buffer, 0, bytemuck::bytes_of(&self.brush_uniforms));

        let dispatch_x = self.grid_width.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = self.grid_height.div_ceil(WORKGROUP_SIZE);

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FFI NCA Compute Encoder"),
                });

        for _ in 0..self.steps_per_frame {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FFI NCA Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.compute_pipeline);
                pass.set_bind_group(0, &self.compute_bind_groups[self.active_state], &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            self.active_state ^= 1;
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn render(&mut self) -> Result<(), (NcaStatus, String)> {
        let Some(surface_bundle) = self.surface_bundle.as_mut() else {
            return Err((
                NcaStatus::Unsupported,
                "render surface is not configured".to_owned(),
            ));
        };

        let Some(render_pipeline) = self.render_pipeline.as_ref() else {
            return Err((
                NcaStatus::Unsupported,
                "render pipeline is not configured".to_owned(),
            ));
        };

        let frame = match surface_bundle.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface_bundle
                    .surface
                    .configure(&self.device, &surface_bundle.config);
                surface_bundle
                    .surface
                    .get_current_texture()
                    .map_err(|err| {
                        (
                            NcaStatus::RuntimeError,
                            format!("failed to acquire surface texture after reconfigure: {err}"),
                        )
                    })?
            }
            Err(err) => {
                return Err((
                    NcaStatus::RuntimeError,
                    format!("failed to acquire surface texture: {err}"),
                ));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FFI NCA Render Encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FFI NCA Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(render_pipeline);
            pass.set_bind_group(0, &self.render_bind_groups[self.active_state], &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn resize_surface(&mut self, width: u32, height: u32) -> Result<(), (NcaStatus, String)> {
        if width == 0 || height == 0 {
            return Err((
                NcaStatus::InvalidArgument,
                "surface size must be greater than zero".to_owned(),
            ));
        }

        let Some(surface_bundle) = self.surface_bundle.as_mut() else {
            return Err((
                NcaStatus::Unsupported,
                "render surface is not configured".to_owned(),
            ));
        };

        surface_bundle.config.width = width;
        surface_bundle.config.height = height;
        surface_bundle
            .surface
            .configure(&self.device, &surface_bundle.config);
        Ok(())
    }

    fn refresh_brush_uniform(&mut self) {
        self.brush_uniforms.flags = [0, 0, 0, 0];

        let mut clear_pending = false;
        if let Some(brush) = self.pending_brush.as_mut() {
            self.brush_uniforms.brush = [
                brush.event.x.clamp(-1.0, 1.0),
                brush.event.y.clamp(-1.0, 1.0),
                brush.event.radius.clamp(0.001, 2.0),
                brush.event.strength.max(0.0),
            ];

            match brush.mode {
                BrushMode::Damage => self.brush_uniforms.flags[0] = 1,
                BrushMode::Growth => self.brush_uniforms.flags[1] = 1,
            }

            if brush.frames_left > 0 {
                brush.frames_left -= 1;
            }
            clear_pending = brush.frames_left == 0;
        }

        if clear_pending {
            self.pending_brush = None;
        }
    }

    #[cfg(windows)]
    fn pump_internal_window_messages(&self) {
        if let Some(window) = &self.internal_window {
            unsafe {
                let mut msg = std::mem::zeroed::<MSG>();
                while PeekMessageW(&mut msg, window.hwnd as HWND, 0, 0, PM_REMOVE) != 0 {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_initial_state(grid_width: u32, grid_height: u32) -> Vec<f32> {
    let cell_count = (grid_width * grid_height) as usize;
    let mut data = vec![0.0_f32; cell_count * STATE_DIM];

    data.par_chunks_mut(STATE_DIM)
        .enumerate()
        .for_each(|(index, cell)| {
            let x = (index as u32) % grid_width;
            let y = (index as u32) / grid_width;

            let fx = (x as f32 / grid_width as f32) * 2.0 - 1.0;
            let fy = 1.0 - (y as f32 / grid_height as f32) * 2.0;
            let r = (fx * fx + fy * fy).sqrt();
            let theta = fy.atan2(fx);
            let noise = pseudo_signed(x, y) * 0.035;

            let core = (1.0 - r * 1.3).clamp(0.0, 1.0);
            let ring = (1.0 - (r - 0.38).abs() * 6.0).clamp(0.0, 1.0);

            cell[0] = (0.65 * core + 0.55 * ring + noise).clamp(0.0, 1.0);
            cell[1] = (0.5 + 0.5 * (9.0 * r).sin() + noise * 0.5).clamp(0.0, 1.0);
            cell[2] = (0.5 + 0.5 * (6.0 * theta).cos() + noise * 0.5).clamp(0.0, 1.0);
            cell[3] = (0.35 * core + 0.85 * ring).clamp(0.0, 1.0);
            cell[4] = (0.5 + 0.5 * (4.0 * theta + r * 8.0).sin()).clamp(0.0, 1.0);
            cell[5] = (0.5 + 0.5 * (5.0 * theta - r * 10.0).cos()).clamp(0.0, 1.0);
            cell[6] = (0.5 + 0.5 * (fx * 11.0).sin() * (fy * 7.0).cos()).clamp(0.0, 1.0);
            cell[7] = (0.3 + 0.7 * core).clamp(0.0, 1.0);
        });

    data
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn pseudo_signed(a: u32, b: u32) -> f32 {
    let h = hash_u32(a.wrapping_mul(0x9e37_79b9) ^ b.wrapping_mul(0x7f4a_7c15));
    let n = (h & 0x000f_ffff) as f32 / 1_048_575.0;
    n * 2.0 - 1.0
}

struct HotReloadWorker {
    stop_tx: Sender<()>,
    thread: JoinHandle<()>,
}

struct NcaEngineHandle {
    core: Mutex<EngineCore>,
    last_error: Mutex<String>,
    // Background file watcher pushes decoded weights through this channel.
    hot_reload: Mutex<Option<HotReloadWorker>>,
    reload_rx: Mutex<Option<Receiver<NetworkParameters>>>,
}

impl NcaEngineHandle {
    fn new(core: EngineCore) -> Self {
        Self {
            core: Mutex::new(core),
            last_error: Mutex::new(String::new()),
            hot_reload: Mutex::new(None),
            reload_rx: Mutex::new(None),
        }
    }

    fn clear_error(&self) {
        if let Ok(mut guard) = self.last_error.lock() {
            guard.clear();
        }
    }

    fn set_error(&self, status: NcaStatus, message: impl Into<String>) -> NcaStatus {
        if let Ok(mut guard) = self.last_error.lock() {
            *guard = message.into();
        }
        status
    }

    fn drain_reloaded_weights(&self) -> Result<(), (NcaStatus, String)> {
        let mut latest_weights = None;

        {
            let rx_guard = self
                .reload_rx
                .lock()
                .map_err(|_| (NcaStatus::RuntimeError, "reload channel lock poisoned".to_owned()))?;
            if let Some(rx) = rx_guard.as_ref() {
                while let Ok(weights) = rx.try_recv() {
                    latest_weights = Some(weights);
                }
            }
        }

        if let Some(weights) = latest_weights {
            let mut core = self
                .core
                .lock()
                .map_err(|_| (NcaStatus::RuntimeError, "engine lock poisoned".to_owned()))?;
            core.upload_weights(&weights)?;
        }

        Ok(())
    }

    fn start_hot_reload(&self, path: PathBuf, debounce_ms: u32) -> Result<(), (NcaStatus, String)> {
        self.stop_hot_reload()?;

        let (weights_tx, weights_rx) = mpsc::channel::<NetworkParameters>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let debounce = Duration::from_millis(debounce_ms.max(10) as u64);

        let thread = thread::spawn(move || {
            run_hot_reload_worker(path, debounce, stop_rx, weights_tx);
        });

        {
            let mut reload_guard = self
                .reload_rx
                .lock()
                .map_err(|_| (NcaStatus::RuntimeError, "reload channel lock poisoned".to_owned()))?;
            *reload_guard = Some(weights_rx);
        }

        {
            let mut hot_reload_guard = self
                .hot_reload
                .lock()
                .map_err(|_| (NcaStatus::RuntimeError, "hot-reload lock poisoned".to_owned()))?;
            *hot_reload_guard = Some(HotReloadWorker { stop_tx, thread });
        }

        Ok(())
    }

    fn stop_hot_reload(&self) -> Result<(), (NcaStatus, String)> {
        let worker = {
            let mut guard = self
                .hot_reload
                .lock()
                .map_err(|_| (NcaStatus::RuntimeError, "hot-reload lock poisoned".to_owned()))?;
            guard.take()
        };

        if let Some(worker) = worker {
            let _ = worker.stop_tx.send(());
            let _ = worker.thread.join();
        }

        let mut rx_guard = self
            .reload_rx
            .lock()
            .map_err(|_| (NcaStatus::RuntimeError, "reload channel lock poisoned".to_owned()))?;
        *rx_guard = None;

        Ok(())
    }
}

fn run_hot_reload_worker(
    path: PathBuf,
    debounce: Duration,
    stop_rx: Receiver<()>,
    weights_tx: Sender<NetworkParameters>,
) {
    // Watch parent directory to handle atomic file replacement workflows.
    let watch_root = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let target_file_name = path
        .file_name()
        .map(|name| name.to_owned())
        .unwrap_or_default();

    let (event_tx, event_rx) = mpsc::channel::<notify::Result<Event>>();

    let mut watcher = match notify::recommended_watcher(move |res| {
        let _ = event_tx.send(res);
    }) {
        Ok(watcher) => watcher,
        Err(err) => {
            log::error!("hot-reload watcher init failed: {err}");
            return;
        }
    };

    if let Err(err) = watcher.watch(&watch_root, RecursiveMode::NonRecursive) {
        log::error!("hot-reload watcher failed for {}: {err}", watch_root.display());
        return;
    }

    let mut last_reload = Instant::now() - debounce;

    loop {
        if stop_rx.try_recv().is_ok() {
            return;
        }

        match event_rx.recv_timeout(Duration::from_millis(120)) {
            Ok(Ok(event)) => {
                if !is_reload_event(&event, &target_file_name) {
                    continue;
                }
                if last_reload.elapsed() < debounce {
                    continue;
                }

                match NetworkParameters::load(&path) {
                    Ok(weights) => {
                        let _ = weights_tx.send(weights);
                        last_reload = Instant::now();
                    }
                    Err(err) => {
                        log::warn!("hot-reload failed to load {}: {}", path.display(), err);
                    }
                }
            }
            Ok(Err(err)) => {
                log::warn!("hot-reload watcher event error: {err}");
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn is_reload_event(event: &Event, target_file_name: &std::ffi::OsStr) -> bool {
    let kind_matches = matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_));
    if !kind_matches {
        return false;
    }

    if event.paths.is_empty() {
        return true;
    }

    event
        .paths
        .iter()
        .any(|path| path.file_name().is_some_and(|name| name == target_file_name))
}

enum SurfaceRequest {
    None,
    HostWindows {
        hwnd: isize,
        hinstance: isize,
        width: u32,
        height: u32,
    },
    InternalWindow {
        width: u32,
        height: u32,
    },
}

struct ParsedConfig {
    grid_width: u32,
    grid_height: u32,
    surface_width: u32,
    surface_height: u32,
    steps_per_frame: u32,
    prefer_mailbox: bool,
    surface_request: SurfaceRequest,
    weights_path: Option<PathBuf>,
}

impl ParsedConfig {
    fn from_ffi(config: &NcaEngineConfig) -> Result<Self, (NcaStatus, String)> {
        let grid_width = if config.grid_width == 0 {
            DEFAULT_GRID_WIDTH
        } else {
            config.grid_width
        };
        let grid_height = if config.grid_height == 0 {
            DEFAULT_GRID_HEIGHT
        } else {
            config.grid_height
        };

        let surface_width = if config.surface_width == 0 {
            DEFAULT_SURFACE_WIDTH
        } else {
            config.surface_width
        };
        let surface_height = if config.surface_height == 0 {
            DEFAULT_SURFACE_HEIGHT
        } else {
            config.surface_height
        };

        let steps_per_frame = if config.steps_per_frame == 0 {
            DEFAULT_STEPS_PER_FRAME
        } else {
            config.steps_per_frame.clamp(1, MAX_STEPS_PER_FRAME)
        };

        let prefer_mailbox = config.prefer_mailbox != 0;

        let surface_request = if config.create_internal_window != 0 {
            SurfaceRequest::InternalWindow {
                width: surface_width,
                height: surface_height,
            }
        } else {
            parse_host_surface_request(&config.host_window, surface_width, surface_height)?
        };

        let weights_path = if config.weights_path.is_null() {
            None
        } else {
            Some(path_from_c_string(config.weights_path)?)
        };

        Ok(Self {
            grid_width,
            grid_height,
            surface_width,
            surface_height,
            steps_per_frame,
            prefer_mailbox,
            surface_request,
            weights_path,
        })
    }
}

fn parse_host_surface_request(
    host: &NcaHostWindowHandle,
    default_width: u32,
    default_height: u32,
) -> Result<SurfaceRequest, (NcaStatus, String)> {
    match host.platform {
        NCA_WINDOW_PLATFORM_NONE => Ok(SurfaceRequest::None),
        NCA_WINDOW_PLATFORM_WINDOWS_HWND => {
            if host.window_handle == 0 {
                return Err((
                    NcaStatus::InvalidArgument,
                    "host window_handle is null for Windows platform".to_owned(),
                ));
            }

            let hwnd = isize::try_from(host.window_handle).map_err(|_| {
                (
                    NcaStatus::InvalidArgument,
                    "host window_handle does not fit into pointer width".to_owned(),
                )
            })?;

            let hinstance = isize::try_from(host.display_handle).unwrap_or(0);
            let width = if host.width == 0 {
                default_width
            } else {
                host.width
            };
            let height = if host.height == 0 {
                default_height
            } else {
                host.height
            };

            Ok(SurfaceRequest::HostWindows {
                hwnd,
                hinstance,
                width,
                height,
            })
        }
        other => Err((
            NcaStatus::Unsupported,
            format!("unsupported host window platform value: {other}"),
        )),
    }
}

fn path_from_c_string(path: *const c_char) -> Result<PathBuf, (NcaStatus, String)> {
    if path.is_null() {
        return Err((NcaStatus::NullPointer, "path pointer is null".to_owned()));
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let value = c_str.to_string_lossy();
    if value.is_empty() {
        return Err((
            NcaStatus::InvalidArgument,
            "path string is empty".to_owned(),
        ));
    }

    Ok(PathBuf::from(value.as_ref()))
}

#[cfg(windows)]
fn create_surface_from_windows_handles(
    instance: &wgpu::Instance,
    hwnd: isize,
    hinstance: isize,
) -> Result<wgpu::Surface<'static>, (NcaStatus, String)> {
    let hwnd_nz = NonZeroIsize::new(hwnd).ok_or_else(|| {
        (
            NcaStatus::InvalidArgument,
            "window handle must be non-zero".to_owned(),
        )
    })?;

    let mut window_handle = Win32WindowHandle::new(hwnd_nz);
    window_handle.hinstance = NonZeroIsize::new(hinstance);

    let display_handle = WindowsDisplayHandle::new();
    let target = wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: RawDisplayHandle::Windows(display_handle),
        raw_window_handle: RawWindowHandle::Win32(window_handle),
    };

    let surface = unsafe { instance.create_surface_unsafe(target) }.map_err(|err| {
        (
            NcaStatus::RuntimeError,
            format!("failed to create surface: {err}"),
        )
    })?;

    Ok(surface)
}

#[cfg(not(windows))]
fn create_surface_from_windows_handles(
    _instance: &wgpu::Instance,
    _hwnd: isize,
    _hinstance: isize,
) -> Result<wgpu::Surface<'static>, (NcaStatus, String)> {
    Err((
        NcaStatus::Unsupported,
        "windows surface handles are not available on this platform".to_owned(),
    ))
}

unsafe fn get_handle<'a>(engine: *mut NcaEngineOpaque) -> Result<&'a NcaEngineHandle, NcaStatus> {
    if engine.is_null() {
        return Err(NcaStatus::NullPointer);
    }
    Ok(&(*engine).handle)
}

#[no_mangle]
pub extern "C" fn nca_engine_create(
    config: *const NcaEngineConfig,
    out_engine: *mut *mut NcaEngineOpaque,
) -> NcaStatus {
    if out_engine.is_null() {
        return NcaStatus::NullPointer;
    }

    unsafe {
        *out_engine = ptr::null_mut();
    }

    if config.is_null() {
        return NcaStatus::NullPointer;
    }

    let parsed = match ParsedConfig::from_ffi(unsafe { &*config }) {
        Ok(parsed) => parsed,
        Err((status, message)) => {
            log::error!("nca_engine_create: invalid config: {}", message);
            return status;
        }
    };

    let network = match parsed.weights_path.as_ref() {
        Some(path) => match NetworkParameters::load(path) {
            Ok(weights) => weights,
            Err(err) => {
                log::error!("nca_engine_create: failed to load weights {}: {}", path.display(), err);
                return NcaStatus::RuntimeError;
            }
        },
        None => NetworkParameters::default_seeded(),
    };

    let core = match pollster::block_on(EngineCore::new(parsed, network)) {
        Ok(core) => core,
        Err((status, message)) => {
            log::error!("nca_engine_create: failed to initialize engine: {}", message);
            return status;
        }
    };

    let boxed = Box::new(NcaEngineOpaque {
        handle: NcaEngineHandle::new(core),
    });

    unsafe {
        // Opaque pointer handoff across FFI boundary.
        *out_engine = Box::into_raw(boxed);
    }

    NcaStatus::Ok
}

#[no_mangle]
pub extern "C" fn nca_engine_destroy(engine: *mut NcaEngineOpaque) {
    if engine.is_null() {
        return;
    }

    // Reclaim ownership that was transferred by Box::into_raw in create().
    let boxed = unsafe { Box::from_raw(engine) };
    let _ = boxed.handle.stop_hot_reload();
}

#[no_mangle]
pub extern "C" fn nca_engine_update(engine: *mut NcaEngineOpaque, delta_seconds: f32) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    handle.clear_error();

    if let Err((status, message)) = handle.drain_reloaded_weights() {
        return handle.set_error(status, message);
    }

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => {
            return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned");
        }
    };

    match core.update(delta_seconds) {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

#[no_mangle]
pub extern "C" fn nca_engine_render(engine: *mut NcaEngineOpaque) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    handle.clear_error();

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => {
            return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned");
        }
    };

    match core.render() {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

#[no_mangle]
pub extern "C" fn nca_engine_resize_surface(
    engine: *mut NcaEngineOpaque,
    width: u32,
    height: u32,
) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    handle.clear_error();

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned"),
    };

    match core.resize_surface(width, height) {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

#[no_mangle]
pub extern "C" fn nca_engine_set_steps_per_frame(
    engine: *mut NcaEngineOpaque,
    steps_per_frame: u32,
) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned"),
    };

    core.set_steps_per_frame(steps_per_frame);
    NcaStatus::Ok
}

#[no_mangle]
pub extern "C" fn nca_engine_load_weights(
    engine: *mut NcaEngineOpaque,
    weights_path: *const c_char,
) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    handle.clear_error();

    let path = match path_from_c_string(weights_path) {
        Ok(path) => path,
        Err((status, message)) => return handle.set_error(status, message),
    };

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned"),
    };

    match core.load_weights_from_path(&path) {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

#[no_mangle]
pub extern "C" fn nca_engine_load_weights_from_memory(
    engine: *mut NcaEngineOpaque,
    data: *const u8,
    data_len: usize,
) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    handle.clear_error();

    if data.is_null() {
        return handle.set_error(NcaStatus::NullPointer, "data pointer is null");
    }
    if data_len == 0 {
        return handle.set_error(NcaStatus::InvalidArgument, "data_len must be greater than zero");
    }

    let bytes = unsafe { std::slice::from_raw_parts(data, data_len) };

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned"),
    };

    match core.load_weights_from_bytes(bytes) {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

fn inject_brush(
    engine: *mut NcaEngineOpaque,
    brush_event: *const NcaBrushEvent,
    mode: BrushMode,
) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    if brush_event.is_null() {
        return handle.set_error(NcaStatus::NullPointer, "brush_event pointer is null");
    }

    let event = unsafe { *brush_event };
    if !event.x.is_finite() || !event.y.is_finite() || !event.radius.is_finite() || !event.strength.is_finite() {
        return handle.set_error(NcaStatus::InvalidArgument, "brush_event contains non-finite values");
    }

    let mut core = match handle.core.lock() {
        Ok(core) => core,
        Err(_) => return handle.set_error(NcaStatus::RuntimeError, "engine lock poisoned"),
    };

    core.queue_brush(mode, event);
    NcaStatus::Ok
}

#[no_mangle]
pub extern "C" fn nca_engine_inject_damage(
    engine: *mut NcaEngineOpaque,
    brush_event: *const NcaBrushEvent,
) -> NcaStatus {
    inject_brush(engine, brush_event, BrushMode::Damage)
}

#[no_mangle]
pub extern "C" fn nca_engine_inject_growth(
    engine: *mut NcaEngineOpaque,
    brush_event: *const NcaBrushEvent,
) -> NcaStatus {
    inject_brush(engine, brush_event, BrushMode::Growth)
}

#[no_mangle]
pub extern "C" fn nca_engine_start_hot_reload(
    engine: *mut NcaEngineOpaque,
    weights_path: *const c_char,
    debounce_ms: u32,
) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    let path = match path_from_c_string(weights_path) {
        Ok(path) => path,
        Err((status, message)) => return handle.set_error(status, message),
    };

    match handle.start_hot_reload(path, debounce_ms) {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

#[no_mangle]
pub extern "C" fn nca_engine_stop_hot_reload(engine: *mut NcaEngineOpaque) -> NcaStatus {
    let handle = match unsafe { get_handle(engine) } {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    match handle.stop_hot_reload() {
        Ok(()) => NcaStatus::Ok,
        Err((status, message)) => handle.set_error(status, message),
    }
}

#[no_mangle]
pub extern "C" fn nca_engine_copy_last_error(
    engine: *const NcaEngineOpaque,
    dst: *mut c_char,
    dst_len: usize,
) -> usize {
    if engine.is_null() {
        return 0;
    }

    let handle = unsafe { &(*engine).handle };
    let message = match handle.last_error.lock() {
        Ok(message) => message.clone(),
        Err(_) => "last_error lock poisoned".to_owned(),
    };

    let required_len = message.len() + 1;

    if dst.is_null() || dst_len == 0 {
        return required_len;
    }

    let copy_len = message.len().min(dst_len - 1);
    unsafe {
        ptr::copy_nonoverlapping(message.as_ptr(), dst as *mut u8, copy_len);
        *dst.add(copy_len) = 0;
    }

    required_len
}

#[no_mangle]
pub extern "C" fn nca_engine_version_major() -> u32 {
    0
}

#[no_mangle]
pub extern "C" fn nca_engine_version_minor() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn nca_engine_version_patch() -> u32 {
    0
}
