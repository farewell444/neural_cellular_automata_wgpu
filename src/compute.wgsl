const STATE_DIM: u32 = 8u;
const ANCHOR_DIM: u32 = 4u;
const INPUT_DIM: u32 = 20u;
const HIDDEN_DIM: u32 = 32u;

struct SimUniforms {
    grid: vec4<u32>,
    params_a: vec4<f32>,
    params_b: vec4<f32>,
};

struct BrushUniforms {
    brush: vec4<f32>,
    flags: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read> state_in: array<f32>;

@group(0) @binding(1)
var<storage, read_write> state_out: array<f32>;

@group(0) @binding(2)
var<uniform> sim: SimUniforms;

@group(0) @binding(3)
var<uniform> brush: BrushUniforms;

@group(0) @binding(4)
var<storage, read> w1: array<f32>;

@group(0) @binding(5)
var<storage, read> b1: array<f32>;

@group(0) @binding(6)
var<storage, read> w2: array<f32>;

@group(0) @binding(7)
var<storage, read> b2: array<f32>;

fn wrap_coord(value: i32, max_value: u32) -> u32 {
    let m = i32(max_value);
    let wrapped = ((value % m) + m) % m;
    return u32(wrapped);
}

fn cell_base(x: u32, y: u32) -> u32 {
    return (y * sim.grid.x + x) * STATE_DIM;
}

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn cell_random(x: u32, y: u32, frame: u32, seed: f32) -> f32 {
    let s = u32(abs(seed) * 10000.0) + frame * 131u;
    let mixed = hash_u32((x + 17u) * 0x9e3779b9u ^ (y + 29u) * 0x7f4a7c15u ^ s);
    return f32(mixed & 0x00ffffffu) / 16777215.0;
}

@compute @workgroup_size(8, 8, 1)
fn nca_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sim.grid.x || gid.y >= sim.grid.y) {
        return;
    }

    let x = gid.x;
    let y = gid.y;

    var center: array<f32, STATE_DIM>;
    let center_base = cell_base(x, y);
    for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
        center[i] = state_in[center_base + i];
    }

    var avg: array<f32, STATE_DIM>;
    for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
        avg[i] = 0.0;
    }

    var alive_max = 0.0;

    for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            let nx = wrap_coord(i32(x) + ox, sim.grid.x);
            let ny = wrap_coord(i32(y) + oy, sim.grid.y);
            let neighbor_base = cell_base(nx, ny);
            alive_max = max(alive_max, state_in[neighbor_base + 0u]);
            for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
                avg[i] = avg[i] + state_in[neighbor_base + i];
            }
        }
    }

    for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
        avg[i] = avg[i] / 9.0;
    }

    let fx = (f32(x) / f32(sim.grid.x)) * 2.0 - 1.0;
    let fy = 1.0 - (f32(y) / f32(sim.grid.y)) * 2.0;
    let p = vec2<f32>(fx, fy);
    let r = length(p);
    let theta = atan2(fy, fx);

    var anchor: array<f32, ANCHOR_DIM>;
    anchor[0] = clamp(1.0 - r * 1.35, 0.0, 1.0);
    anchor[1] = 0.5 + 0.5 * sin(11.0 * r - sim.params_a.w * 1.1);
    anchor[2] = 0.5 + 0.5 * cos(7.0 * theta + sim.params_a.w * 0.8);
    anchor[3] = clamp(1.0 - abs(r - 0.38) * 6.0, 0.0, 1.0);

    var input: array<f32, INPUT_DIM>;
    for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
        input[i] = center[i];
        input[STATE_DIM + i] = center[i] + (avg[i] - center[i]) * sim.params_a.z;
    }
    for (var i: u32 = 0u; i < ANCHOR_DIM; i = i + 1u) {
        input[STATE_DIM * 2u + i] = anchor[i] * sim.params_a.y;
    }

    var hidden: array<f32, HIDDEN_DIM>;
    for (var h: u32 = 0u; h < HIDDEN_DIM; h = h + 1u) {
        var sum = b1[h];
        for (var i: u32 = 0u; i < INPUT_DIM; i = i + 1u) {
            sum = sum + input[i] * w1[h * INPUT_DIM + i];
        }
        hidden[h] = tanh(sum);
    }

    var next_state: array<f32, STATE_DIM>;
    let gate = select(
        0.0,
        1.0,
        cell_random(x, y, sim.grid.w, sim.params_b.w) <= sim.params_b.x,
    );
    for (var o: u32 = 0u; o < STATE_DIM; o = o + 1u) {
        var sum = b2[o];
        for (var h: u32 = 0u; h < HIDDEN_DIM; h = h + 1u) {
            sum = sum + hidden[h] * w2[o * HIDDEN_DIM + h];
        }
        let delta = clamp(tanh(sum), -sim.params_b.y, sim.params_b.y);
        next_state[o] = clamp(center[o] + sim.params_a.x * delta * gate, 0.0, 1.0);
    }

    let is_alive = alive_max > sim.params_b.z || anchor[0] > 0.04;
    if (!is_alive) {
        for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
            next_state[i] = max(next_state[i] - 0.01, 0.0);
        }
    }

    if (brush.flags.x == 1u) {
        let dx = fx - brush.brush.x;
        let dy = fy - brush.brush.y;
        let dist = length(vec2<f32>(dx, dy));
        if (dist < brush.brush.z) {
            let falloff = 1.0 - dist / brush.brush.z;
            let destruction = clamp(1.0 - brush.brush.w * falloff, 0.0, 1.0);
            for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
                next_state[i] = next_state[i] * destruction;
            }
        }
    }

    if (brush.flags.y == 1u) {
        let dx = fx - brush.brush.x;
        let dy = fy - brush.brush.y;
        let dist = length(vec2<f32>(dx, dy));
        if (dist < brush.brush.z) {
            let falloff = 1.0 - dist / brush.brush.z;
            let growth = brush.brush.w * falloff;
            for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
                let anchor_boost = anchor[i % ANCHOR_DIM];
                next_state[i] = clamp(next_state[i] + growth * (0.2 + 0.6 * anchor_boost), 0.0, 1.0);
            }
        }
    }

    let out_base = cell_base(x, y);
    for (var i: u32 = 0u; i < STATE_DIM; i = i + 1u) {
        state_out[out_base + i] = next_state[i];
    }
}
