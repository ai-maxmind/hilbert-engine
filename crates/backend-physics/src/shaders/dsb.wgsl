// dsb.wgsl - Discrete Simulated Bifurcation Compute Shader
// Optimized branchless CSR SpMV and Symplectic Euler Integrator for N coupled oscillators across R replicas.

struct SimParams {
    num_spins: u32,
    num_replicas: u32,
    dt: f32,
    a0: f32,
    c0: f32,
    num_steps: u32,
    current_step: u32,
    mode: u32, // 0 = dSB (discrete), 1 = bSB (ballistic), 2 = cSB (chaotic)
    num_x: u32, // Number of Minimizer variables [0..num_x). Remaining [num_x..num_spins) are Maximizers.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Bind group 0: Static Problem Definition (CSR Matrix and Linear Field)
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> h: array<f32>;

// Bind group 1: Dynamic Simulation State
@group(1) @binding(0) var<storage, read_write> params: SimParams;
@group(1) @binding(1) var<storage, read_write> x: array<f32>;
@group(1) @binding(2) var<storage, read_write> y: array<f32>;
@group(1) @binding(3) var<storage, read_write> spins: array<i32>;

// Entry point 1: update_y
// Computes local interaction field via CSR SpMV and updates momentum y
@compute @workgroup_size(64, 1, 1)
fn update_y(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let spin_id = global_id.x;
    let replica_id = global_id.y;

    if (spin_id >= params.num_spins || replica_id >= params.num_replicas) {
        return;
    }

    let replica_offset = replica_id * params.num_spins;
    let idx = replica_offset + spin_id;

    // Time schedule: a_t = a0 * (step / M)
    let step = f32(params.current_step);
    let m = f32(params.num_steps);
    let a_t = params.a0 * (step / m);
    let detuning = -(1.0f - a_t);

    // Branchless SpMV over CSR row of spin_id
    let start = row_ptr[spin_id];
    let end = row_ptr[spin_id + 1u];

    var field: f32 = h[spin_id];
    for (var k = start; k < end; k = k + 1u) {
        let col = col_indices[k];
        let neighbor_x = x[replica_offset + col];
        // Branchless sign extraction: +1.0 if x >= 0.0 else -1.0
        let s_j = select(-1.0f, 1.0f, neighbor_x >= 0.0f);
        field = field + values[k] * s_j;
    }

    // Chaotic perturbation for cSB (Ergodic resonance to escape spin-glass metastable states)
    if (params.mode == 2u) {
        let phi = fract(sin(f32(spin_id) * 12.9898f + f32(replica_id) * 78.233f) * 43758.5453f) * 6.2831853f;
        let decay = max(0.0f, 1.0f - (step / m));
        let chaos_force = 0.25f * decay * sin(25.0f * (step / m) + phi);
        field = field + chaos_force;
    }

    // Adversarial Minimax dynamics:
    // If spin_id < params.num_x: Minimizer (moves along -grad H, force_sign = +1.0)
    // If spin_id >= params.num_x: Maximizer (moves along +grad H, force_sign = -1.0)
    let is_minimizer = spin_id < params.num_x;
    let force_sign = select(-1.0f, 1.0f, is_minimizer);

    // Momentum update: dy = [ -(1 - a_t)*x + c0*field * force_sign ] * dt
    let dy = (detuning * x[idx] + params.c0 * field * force_sign) * params.dt;
    y[idx] = y[idx] + dy;
}

// Entry point 2: update_x
// Updates position x and applies boundary conditions (dSB inelastic or bSB/cSB ballistic reflection)
@compute @workgroup_size(64, 1, 1)
fn update_x(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let spin_id = global_id.x;
    let replica_id = global_id.y;

    if (spin_id >= params.num_spins || replica_id >= params.num_replicas) {
        return;
    }

    let idx = replica_id * params.num_spins + spin_id;

    // Position update: dx = y * dt
    let new_x = x[idx] + params.dt * y[idx];

    // Boundary condition
    let over_wall = abs(new_x) > 1.0f;
    let clamped_x = select(-1.0f, 1.0f, new_x >= 0.0f);

    x[idx] = select(new_x, clamped_x, over_wall);

    if (params.mode == 0u) {
        // dSB: Inelastic hard-wall (zero out momentum on impact)
        y[idx] = select(y[idx], 0.0f, over_wall);
    } else {
        // bSB / cSB: Ballistic bouncing with damping factor (-0.5) to escape flat boundary traps
        y[idx] = select(y[idx], -0.5f * y[idx], over_wall);
    }
}

// Entry point 3: advance_time
// Increments simulation step counter in GPU storage buffer
@compute @workgroup_size(1, 1, 1)
fn advance_time(@builtin(global_invocation_id) global_id: vec3<u32>) {
    params.current_step = params.current_step + 1u;
}

// Readout entry point: Binarizes continuous x coordinates to discrete spins s_i in {-1, +1}
@compute @workgroup_size(64, 1, 1)
fn readout_spins(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let spin_id = global_id.x;
    let replica_id = global_id.y;

    if (spin_id >= params.num_spins || replica_id >= params.num_replicas) {
        return;
    }

    let idx = replica_id * params.num_spins + spin_id;
    spins[idx] = select(-1, 1, x[idx] >= 0.0f);
}
