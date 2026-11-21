//! Module `gpu` implements GPU-accelerated Discrete Simulated Bifurcation (dSB)
//! using `wgpu` and WGSL Compute Shaders.
//!
//! # Architecture & GPU Optimizations:
//!
//! 1. **Zero-Copy Host-to-Device:**
//!    All CSR arrays (`row_ptr`, `col_indices`, `values`, `h`) are cast directly to GPU
//!    storage buffers using `bytemuck::cast_slice` without memory copies.
//!
//! 2. **Single CommandEncoder Dispatch:**
//!    All `max_steps` time-integration cycles (consisting of SpMV momentum updates,
//!    position updates, and time advancements) are recorded into a single `wgpu::CommandEncoder`.
//!    This eliminates all CPU-GPU synchronization stalls and PCIe turnaround delays.
//!
//! 3. **Branchless Massive Parallelism:**
//!    Workgroups of size 64 process thousands of independent replicas simultaneously
//!    on the 2D compute grid: `(num_spins / 64, num_replicas)`.

use std::sync::mpsc;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::dsb::{DsbConfig, DsbResult};
use crate::ising::{IsingCsr, IsingError};

/// GPU simulation parameters mapped byte-for-byte to WGSL `SimParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub num_spins: u32,
    pub num_replicas: u32,
    pub dt: f32,
    pub a0: f32,
    pub c0: f32,
    pub num_steps: u32,
    pub current_step: u32,
    pub mode: u32,
    pub num_x: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Errors occurring during GPU initialization, execution, or readback.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("Ising model error: {0}")]
    Ising(#[from] IsingError),

    #[error("No suitable GPU adapter found")]
    NoAdapter,

    #[error("Failed to request GPU device: {0}")]
    DeviceRequestFailed(String),

    #[error("GPU buffer mapping error: {0}")]
    BufferMapFailed(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Reusable GPU context caching the `wgpu::Device` and `wgpu::Queue`.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize a new GPU context with high performance preference.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|| {
            // Fallback if primary high performance adapter is unavailable
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dsb_gpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceRequestFailed(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }
}

/// Solve the Ising problem using GPU-accelerated Discrete Simulated Bifurcation.
pub fn solve_dsb_gpu(ising: &IsingCsr, config: &DsbConfig) -> Result<DsbResult, GpuError> {
    let ctx = GpuContext::new()?;
    solve_dsb_gpu_with_context(&ctx, ising, config)
}

/// Solve the Ising problem using an existing GPU context.
pub fn solve_dsb_gpu_with_context(
    ctx: &GpuContext,
    ising: &IsingCsr,
    config: &DsbConfig,
) -> Result<DsbResult, GpuError> {
    if config.num_steps == 0 {
        return Err(GpuError::Config("num_steps must be > 0".into()));
    }
    if config.num_replicas == 0 {
        return Err(GpuError::Config("num_replicas must be > 0".into()));
    }

    let n = ising.num_spins;
    if n == 0 {
        return Ok(DsbResult {
            best_spins: Vec::new(),
            best_energy: 0.0,
            steps_executed: config.num_steps,
            c0: 0.0,
            replica_spins: vec![Vec::new(); config.num_replicas],
            replica_energies: vec![0.0; config.num_replicas],
        });
    }

    let c0 = config.c0.unwrap_or_else(|| ising.compute_c0());
    let device = &ctx.device;
    let queue = &ctx.queue;

    // 1. Prepare Bind Group 0: Problem Definition Buffers
    let row_ptr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("row_ptr_buf"),
        contents: bytemuck::cast_slice(&ising.row_ptr),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let col_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("col_indices_buf"),
        contents: bytemuck::cast_slice(&ising.col_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let values_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("values_buf"),
        contents: bytemuck::cast_slice(&ising.values),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let h_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("h_buf"),
        contents: bytemuck::cast_slice(&ising.h),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 2. Prepare Bind Group 1: Simulation State Buffers
    let total_elements = n * config.num_replicas;
    let mut init_x = vec![0.0f32; total_elements];
    let mut init_y = vec![0.0f32; total_elements];

    // Seed PRNG deterministically according to DST requirements
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);
    for r in 0..config.num_replicas {
        let offset = r * n;
        for i in 0..n {
            init_x[offset + i] = rng.gen_range(-0.1f32..0.1f32);
            init_y[offset + i] = rng.gen_range(-0.1f32..0.1f32);
        }
    }

    let num_x = config.num_x.unwrap_or(n) as u32;

    let sim_params = SimParams {
        num_spins: n as u32,
        num_replicas: config.num_replicas as u32,
        dt: config.dt,
        a0: config.a0,
        c0,
        num_steps: config.num_steps as u32,
        current_step: 0,
        mode: config.mode as u32,
        num_x,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params_buf"),
        contents: bytemuck::bytes_of(&sim_params),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x_buf"),
        contents: bytemuck::cast_slice(&init_x),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("y_buf"),
        contents: bytemuck::cast_slice(&init_y),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let spins_byte_size = (total_elements * std::mem::size_of::<i32>()) as u64;

    let spins_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spins_buf"),
        size: spins_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buf"),
        size: spins_byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Compile WGSL Shader Module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dsb_compute_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dsb.wgsl").into()),
    });

    // 4. Create Bind Group Layouts
    let bind_group_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_problem"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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

    let bind_group_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_simulation"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dsb_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
        push_constant_ranges: &[],
    });

    // 5. Create Compute Pipelines for the 3 core entry points + readout
    let pipeline_update_y = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_update_y"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("update_y"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pipeline_update_x = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_update_x"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("update_x"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pipeline_advance_time = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_advance_time"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("advance_time"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pipeline_readout = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_readout"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("readout_spins"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // 6. Instantiate Bind Groups
    let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group_problem"),
        layout: &bind_group_layout_0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: row_ptr_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: col_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: h_buf.as_entire_binding(),
            },
        ],
    });

    let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group_sim"),
        layout: &bind_group_layout_1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: spins_buf.as_entire_binding(),
            },
        ],
    });

    // 7. Record ALL simulation steps into a SINGLE CommandEncoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dsb_single_encoder"),
    });

    let workgroups_x = ((n as u32) + 63) / 64;
    let workgroups_y = config.num_replicas as u32;

    for _step in 0..config.num_steps {
        // Pass 1: update_y
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass_update_y"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_update_y);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 2: update_x
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass_update_x"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_update_x);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 3: advance_time
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass_advance_time"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_advance_time);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    // Final Pass: Readout continuous positions into discrete {-1, +1} spins
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass_readout"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_readout);
        pass.set_bind_group(0, &bind_group_0, &[]);
        pass.set_bind_group(1, &bind_group_1, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // Zero-overhead GPU copy from simulation buffer to staging readback buffer
    encoder.copy_buffer_to_buffer(&spins_buf, 0, &staging_buf, 0, spins_byte_size);

    // Submit complete command stream in one call!
    queue.submit(std::iter::once(encoder.finish()));

    // 8. Map staging buffer and retrieve spins
    let buffer_slice = staging_buf.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });

    let _ = device.poll(wgpu::Maintain::wait());
    receiver
        .recv()
        .map_err(|e| GpuError::BufferMapFailed(e.to_string()))?
        .map_err(|e| GpuError::BufferMapFailed(e.to_string()))?;

    let mapped_view = buffer_slice.get_mapped_range();
    let flat_spins: &[i32] = bytemuck::cast_slice(&mapped_view);

    // 9. Decode spins per replica and calculate exact Hamiltonian energies
    let mut replica_spins: Vec<Vec<i8>> = Vec::with_capacity(config.num_replicas);
    let mut replica_energies: Vec<f64> = Vec::with_capacity(config.num_replicas);

    let mut best_spins: Vec<i8> = Vec::new();
    let mut best_energy = f64::INFINITY;
    let mut best_deviations = usize::MAX;
    let mut best_violation = f64::INFINITY;

    for r in 0..config.num_replicas {
        let offset = r * n;
        let mut spins = Vec::with_capacity(n);
        for i in 0..n {
            let s = if flat_spins[offset + i] >= 0 { 1i8 } else { -1i8 };
            spins.push(s);
        }
        let energy = ising.compute_energy(&spins)?;

        if let Some(num_x) = config.num_x {
            let (violation, deviations) = ising.compute_saddle_violation(num_x, &spins);
            if deviations < best_deviations
                || (deviations == best_deviations && (violation < best_violation || (violation == best_violation && energy < best_energy)))
            {
                best_deviations = deviations;
                best_violation = violation;
                best_energy = energy;
                best_spins = spins.clone();
            }
        } else {
            if energy < best_energy {
                best_energy = energy;
                best_spins = spins.clone();
            }
        }

        replica_spins.push(spins);
        replica_energies.push(energy);
    }

    drop(mapped_view);
    staging_buf.unmap();

    Ok(DsbResult {
        best_spins,
        best_energy,
        steps_executed: config.num_steps,
        c0,
        replica_spins,
        replica_energies,
    })
}
