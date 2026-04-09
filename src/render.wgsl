const STATE_DIM: u32 = 8u;

struct SimUniforms {
    grid: vec4<u32>,
    params_a: vec4<f32>,
    params_b: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> state_cells: array<f32>;

@group(0) @binding(1)
var<uniform> sim: SimUniforms;

fn cell_base(x: u32, y: u32) -> u32 {
    return (y * sim.grid.x + x) * STATE_DIM;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0)
    );

    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0)
    );

    var out: VSOut;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    let x = min(u32(floor(uv.x * f32(sim.grid.x))), sim.grid.x - 1u);
    let y = min(u32(floor((1.0 - uv.y) * f32(sim.grid.y))), sim.grid.y - 1u);

    let base = cell_base(x, y);
    let c0 = state_cells[base + 0u];
    let c1 = state_cells[base + 1u];
    let c2 = state_cells[base + 2u];
    let c3 = state_cells[base + 3u];
    let c4 = state_cells[base + 4u];

    let pulse = 0.5 + 0.5 * sin(sim.params_a.w * 2.0 + c1 * 6.28318);
    let glow = clamp(c0 * 1.2 + c3 * 0.8, 0.0, 1.0);

    let color = vec3<f32>(
        0.08 + 0.85 * c0 + 0.25 * c4,
        0.05 + 0.75 * c1 * pulse,
        0.10 + 0.90 * c2
    ) * (0.35 + 0.75 * glow);

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
