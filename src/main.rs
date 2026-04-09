mod model;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use model::{NetworkParameters, STATE_DIM};
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const GRID_WIDTH: u32 = 512;
const GRID_HEIGHT: u32 = 512;
const WORKGROUP_SIZE: u32 = 8;
const DEFAULT_STEPS_PER_FRAME: u32 = 4;
const MAX_STEPS_PER_FRAME: u32 = 16;

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

#[derive(Debug, Clone)]
struct RuntimeConfig {
    weights_path: PathBuf,
    export_default_weights: Option<PathBuf>,
    steps_per_frame: u32,
}

impl RuntimeConfig {
    fn parse() -> Self {
        let mut weights_path = PathBuf::from(
            std::env::var("CELLFORGE_WEIGHTS")
                .or_else(|_| std::env::var("NCA_WEIGHTS"))
                .unwrap_or_else(|_| "assets/nca_weights.bin".to_owned()),
        );
        let mut export_default_weights = None;
        let mut steps_per_frame = DEFAULT_STEPS_PER_FRAME;

        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--weights" => {
                    if let Some(path) = args.next() {
                        weights_path = PathBuf::from(path);
                    } else {
                        log::warn!("--weights requires a path argument");
                    }
                }
                "--steps" => {
                    if let Some(value) = args.next() {
                        match value.parse::<u32>() {
                            Ok(parsed) => {
                                steps_per_frame = parsed.clamp(1, MAX_STEPS_PER_FRAME);
                            }
                            Err(_) => {
                                log::warn!("invalid value for --steps: {value}");
                            }
                        }
                    } else {
                        log::warn!("--steps requires an integer argument");
                    }
                }
                "--export-default-weights" => {
                    let export_path = match args.peek() {
                        Some(next) if !next.starts_with("--") => {
                            PathBuf::from(args.next().expect("peeked value must exist"))
                        }
                        _ => weights_path.clone(),
                    };
                    export_default_weights = Some(export_path);
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                unknown => {
                    log::warn!("unknown argument: {unknown}");
                }
            }
        }

        Self {
            weights_path,
            export_default_weights,
            steps_per_frame,
        }
    }
}

fn print_usage() {
    println!(
        "Usage:\n  cargo run --release -- [OPTIONS]\n\nOptions:\n  --weights <PATH>                 Load CellForge weights from a .bin file\n  --export-default-weights [PATH]  Export seeded baseline weights and exit\n  --steps <N>                      Compute updates per frame (1..={MAX_STEPS_PER_FRAME})\n  -h, --help                       Show this message\n\nEnv:\n  CELLFORGE_WEIGHTS                Preferred default path for --weights\n  NCA_WEIGHTS                      Legacy alias for CELLFORGE_WEIGHTS (fallback: assets/nca_weights.bin)"
    );
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

fn build_initial_state() -> Vec<f32> {
    let cell_count = (GRID_WIDTH * GRID_HEIGHT) as usize;
    let mut data = vec![0.0_f32; cell_count * STATE_DIM];

    // Use CPU-side data parallelism for initialization before handing all updates to the GPU.
    data.par_chunks_mut(STATE_DIM)
        .enumerate()
        .for_each(|(index, cell)| {
            let x = (index as u32) % GRID_WIDTH;
            let y = (index as u32) / GRID_WIDTH;

            let fx = (x as f32 / GRID_WIDTH as f32) * 2.0 - 1.0;
            let fy = 1.0 - (y as f32 / GRID_HEIGHT as f32) * 2.0;
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

struct App {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    compute_bind_groups: [wgpu::BindGroup; 2],
    render_bind_groups: [wgpu::BindGroup; 2],
    _state_buffers: [wgpu::Buffer; 2],
    _network_buffers: [wgpu::Buffer; 4],
    sim_buffer: wgpu::Buffer,
    brush_buffer: wgpu::Buffer,
    sim_uniforms: SimUniforms,
    brush_uniforms: BrushUniforms,
    active_state: usize,
    steps_per_frame: u32,
    mouse_ndc: [f32; 2],
    mouse_down: bool,
    start_time: Instant,
    _window_guard: Arc<winit::window::Window>,
}

impl App {
    async fn new(
        window: Arc<winit::window::Window>,
        network: NetworkParameters,
        steps_per_frame: u32,
    ) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapters found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("CellForge Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("failed to request device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let initial_state = build_initial_state();

        let state_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("State Buffer A"),
            contents: bytemuck::cast_slice(&initial_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let state_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("State Buffer B"),
            contents: bytemuck::cast_slice(&initial_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sim_uniforms = SimUniforms {
            grid: [GRID_WIDTH, GRID_HEIGHT, STATE_DIM as u32, 0],
            // params_a: dt, anchor_gain, neighbor_mix, time_seconds
            params_a: [0.18, 0.90, 0.62, 0.0],
            // params_b: fire_rate, max_delta, alive_threshold, noise_seed
            params_b: [0.58, 0.35, 0.08, 0.37],
        };

        let brush_uniforms = BrushUniforms {
            brush: [0.0, 0.0, 0.12, 1.0],
            flags: [0, 0, 0, 0],
        };

        let sim_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Uniform Buffer"),
            contents: bytemuck::bytes_of(&sim_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let brush_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brush Uniform Buffer"),
            contents: bytemuck::bytes_of(&brush_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let w1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("W1 Buffer"),
            contents: bytemuck::cast_slice(&network.w1),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B1 Buffer"),
            contents: bytemuck::cast_slice(&network.b1),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let w2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("W2 Buffer"),
            contents: bytemuck::cast_slice(&network.w2),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B2 Buffer"),
            contents: bytemuck::cast_slice(&network.b2),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
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
            label: Some("Compute Bind Group A->B"),
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
            label: Some("Compute Bind Group B->A"),
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
            label: Some("Render Bind Group A"),
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
            label: Some("Render Bind Group B"),
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
            label: Some("CellForge Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CellForge Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CellForge Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "nca_step",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CellForge Render Pipeline"),
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
                    format: config.format,
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
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            compute_pipeline,
            render_pipeline,
            compute_bind_groups: [compute_bind_group_0, compute_bind_group_1],
            render_bind_groups: [render_bind_group_a, render_bind_group_b],
            _state_buffers: [state_a, state_b],
            _network_buffers: [w1_buffer, b1_buffer, w2_buffer, b2_buffer],
            sim_buffer,
            brush_buffer,
            sim_uniforms,
            brush_uniforms,
            active_state: 0,
            steps_per_frame,
            mouse_ndc: [0.0, 0.0],
            mouse_down: false,
            start_time: Instant::now(),
            _window_guard: window,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn update_mouse_position(&mut self, position: PhysicalPosition<f64>) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }

        let x = (position.x as f32 / self.size.width as f32) * 2.0 - 1.0;
        let y = 1.0 - (position.y as f32 / self.size.height as f32) * 2.0;
        self.mouse_ndc = [x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0)];
    }

    fn update_uniforms(&mut self) {
        self.sim_uniforms.grid[3] = self.sim_uniforms.grid[3].wrapping_add(1);
        self.sim_uniforms.params_a[3] = self.start_time.elapsed().as_secs_f32();
        self.brush_uniforms.brush[0] = self.mouse_ndc[0];
        self.brush_uniforms.brush[1] = self.mouse_ndc[1];
        self.brush_uniforms.flags[0] = u32::from(self.mouse_down);

        self.queue
            .write_buffer(&self.sim_buffer, 0, bytemuck::bytes_of(&self.sim_uniforms));
        self.queue
            .write_buffer(&self.brush_buffer, 0, bytemuck::bytes_of(&self.brush_uniforms));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_uniforms();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CellForge Command Encoder"),
                });

        let dispatch_x = GRID_WIDTH.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = GRID_HEIGHT.div_ceil(WORKGROUP_SIZE);

        for _ in 0..self.steps_per_frame {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CellForge Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.compute_pipeline);
                pass.set_bind_group(0, &self.compute_bind_groups[self.active_state], &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            self.active_state ^= 1;
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CellForge Render Pass"),
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
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bind_groups[self.active_state], &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();

    let runtime = RuntimeConfig::parse();

    if let Some(path) = runtime.export_default_weights.clone() {
        let defaults = NetworkParameters::default_seeded();
        match defaults.save(&path) {
            Ok(()) => {
                println!("Exported baseline CellForge weights to {}", path.display());
                return;
            }
            Err(err) => {
                panic!("failed to export baseline weights to {}: {err}", path.display());
            }
        }
    }

    let network = NetworkParameters::load_or_default(&runtime.weights_path);

    log::info!(
        "starting CellForge runtime | weights: {} | steps/frame: {}",
        runtime.weights_path.display(),
        runtime.steps_per_frame
    );

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("CellForge Engine · WGPU")
            .with_inner_size(PhysicalSize::new(1200, 800))
            .with_resizable(true)
            .build(&event_loop)
            .expect("failed to create window"),
    );

    let mut app = pollster::block_on(App::new(
        window.clone(),
        network,
        runtime.steps_per_frame,
    ));

    event_loop
        .run(move |event, event_loop_window_target| {
            event_loop_window_target.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                    WindowEvent::CloseRequested => {
                        event_loop_window_target.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        app.resize(new_size);
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        app.resize(window.inner_size());
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        app.update_mouse_position(position);
                    }
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    } => {
                        app.mouse_down = state == ElementState::Pressed;
                    }
                    WindowEvent::RedrawRequested => match app.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            app.resize(app.size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            event_loop_window_target.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            log::warn!("surface timeout");
                        }
                    },
                    _ => {}
                },
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop error");
}
