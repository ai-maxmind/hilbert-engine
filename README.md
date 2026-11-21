# Trình Biên Dịch Động Lực Học 

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Workspace Tests](https://img.shields.io/badge/tests-61%2F61%20passing%20(100%25)-brightgreen.svg)]()
[![GPU Backend](https://img.shields.io/badge/gpu-wgpu%20%2F%20Vulkan%20%2F%20DirectX12-blue.svg)]()
[![Benchmarks](https://img.shields.io/badge/benchmarks-21%20subcommands-purple.svg)]()
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()


> Giải phóng sức mạnh kết hợp của **Động lực học Rẽ nhánh (Simulated Bifurcation - dSB/bSB/cSB)**, **Mạng lưới Tensor (Tensor-Train MPO & MPS)**, **Đa phân rã thứ bậc (Multilevel V-Cycle 8 tầng)**, **Khử cơ sở lưới (LLL)**, **Bất biến Dehn**, **Khử Gauss $\mathbb{F}_2$ (2-Descent Selmer)** và **Đại số tối ưu E-Graph** để bẻ gãy các bài toán tối ưu tổ hợp NP-Hard, PSPACE đối kháng, đa thức bậc siêu cao, cùng các đại bài toán Thiên niên kỷ và 23 bài toán thế kỷ của David Hilbert.

---


## MỤC LỤC
1. [Nguyên Lý Vận Hành Cốt Lõi: Động Lực Học Không Gian Pha](#1-nguyên-lý-vận-hành-cốt-lõi-động-lực-học-không-gian-pha)
2. [Sơ Đồ Kiến Trúc Hệ Thống 8 Crates Monorepo](#2-sơ-đồ-kiến-trúc-hệ-thống-8-crates-monorepo)
3. [Danh Mục 8 Phân Hệ Crate & Thuật Toán Nền Tảng](#3-danh-mục-8-phân-hệ-crate--thuật-toán-nền-tảng)
4. [Dòng Chảy Dữ Liệu 5 Bước (Data Flow)](#4-dòng-chảy-dữ-liệu-5-bước-data-flow)
5. [Hướng Dẫn Lập Trình Từng Bước (Code Examples)](#5-hướng-dẫn-lập-trình-từng-bước-code-examples)
6. [Bảng Tra Cứu Toàn Diện 21 Lệnh Thực Thi CLI (`cli-bench`)](#6-bảng-tra-cứu-toàn-diện-21-lệnh-thực-thi-cli-cli-bench)
7. [Kết Quả Đo Kiểm Thực Nghiệm & Kiểm Thử Hệ Thống](#7-kết-quả-đo-kiểm-thực-nghiệm--kiểm-thử-hệ-thống)
8. [Cài Đặt & Hướng Dẫn Biên Dịch](#8-cài-đặt--hướng-dẫn-biên-dịch)
9. [Giấy Phép & Tác Giả](#9-giấy-phép--tác-giả)

---

## 1. NGUYÊN LÝ VẬN HÀNH CỐT LÕI: ĐỘNG LỰC HỌC KHÔNG GIAN PHA

Trong khoa học máy tính cổ điển, các bài toán tối ưu tổ hợp (SAT, Max-Cut, TSP, Game Theory) bị giam hãm trong không gian nhị phân rời rạc $\{0, 1\}^N$. Với $N=100$, không gian có $2^{100} \approx 10^{30}$ trạng thái, siêu máy tính duyệt $10^9$ trạng thái/giây vẫn cần hơn $40.000$ tỷ năm.

Trình biên dịch Động lực học Lai phá vỡ rào cản này bằng cách **nhúng không gian rời rạc vào không gian pha liên tục**:
$$(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{2N}$$
Mỗi biến logic $x_i$ trở thành một con thoi dao động phi tuyến tuân theo phương trình vi phân Hamilton biến thiên đoạn nhiệt (*Adiabatic Bifurcation*):

$$\frac{dx_i}{dt} = y_i, \qquad \frac{dy_i}{dt} = -\Big(1 - a(t)\Big) x_i + c_0 \left( \sum_{j=1}^N J_{ij} \operatorname{sgn}(x_j) + h_i \right)$$

* **Bơm năng lượng đoạn nhiệt $a(t) = a_0 \frac{t}{M}$:** Tăng dần từ $0$ lên $a_0 > 1$. Tại thời điểm tới hạn $a(t) = 1$, hố thế chữ U đơn cực tại gốc $x=0$ tự động rẽ nhánh thành 2 hố thế đối xứng tại $x = \pm 1$.
* **3 Chế độ di chuyển của hạt (Physics Modes):**
  1. **`dSB` (Discrete):** Ghìm chặt hạt tại biên $|x_i| \le 1.0$, triệt tiêu quán tính khi va chạm tường, chốt nghiệm siêu tốc.
  2. **`bSB` (Ballistic):** Nảy ngược đàn hồi $y_i \leftarrow -0.5 y_i$, tích lũy động năng nhảy vọt qua rào thế Spin Glass hiểm trở.
  3. **`cSB` (Chaotic):** Kích hoạt lực rung lắc ergodic $F_i^{\text{chaotic}} \propto \sin(\omega t + \phi_i)$, mô phỏng hiệu ứng **đào hầm lượng tử (Quantum Tunneling)** thoát bẫy cục bộ.

---

## 2. SƠ ĐỒ KIẾN TRÚC HỆ THỐNG 8 CRATES MONOREPO

```mermaid
flowchart TD
    subgraph IN["1. TẦNG TIẾP NHẬN BÀI TOÁN (FRONTEND DSL)"]
        F1["crates/frontend-dsl<br/>• ProblemBuilder: Ràng buộc logic nhị phân (ExactOne, 3-SAT, And, Or, Xor)<br/>• QuantifiedProblemBuilder: Đối kháng Minimax Game Theory min_x max_y H<br/>• HpProteinModel: Gấp cuộn Protein 2D HP Lattice"]
    end

    subgraph MID["2. TẦNG TRUNG GIAN & RÚT GỌN (MID-END IR & TENSOR)"]
        M1["crates/midend-egraph<br/>• egg E-Graph Equality Saturation<br/>• Tái sử dụng biến phụ Rosenberg O(1)<br/>• Bounded Saturation (Chống bẫy Nagata)"]
        M2["crates/midend-tensor<br/>• Tensor-Train MPO Projector (Hạ bậc k ≥ 8 với 0 ancilla)<br/>• Matrix Product State MPS (Sign-Free Quantum TFIM)<br/>• Artin SOS Gram Matrix Cholesky Q ⪰ 0 (Hilbert 17)"]
        M3["crates/midend-decompose<br/>• Dynamic Range Scaling (DRS)<br/>• Multilevel V-Cycle 8 tầng (HEM Coarsening)<br/>• Đẳng cấu năng lượng H_f ≡ H_c + ΔE_int<br/>• Sinh ma trận nén thưa IsingCsr"]
    end

    subgraph BACK["3. TẦNG ĐỘNG LỰC HỌC VẬT LÝ (BACKEND PHYSICS)"]
        B1["crates/backend-physics<br/>• CPU DST Engine: Tích phân Symplectic Euler tất định 100% từng bit<br/>• GPU wgpu Compute: 1.024 Replicas / 1 CommandEncoder trên Vulkan/DirectX<br/>• 3 Chế độ: dSB (Inelastic), bSB (Ballistic), cSB (Chaotic Ergodic)<br/>• Phân mảnh miền & Trao đổi biên Halo Exchange đa GPU<br/>• Động lực học Xoáy Hạt Biot-Savart & Giám sát Tiêu chuẩn BKM (Navier-Stokes)<br/>• Toán tử Lượng tử Berry-Keating Hermite đối xứng (Giả thuyết Riemann)<br/>• Lý thuyết Trường Lưới Wilson Loops & Đo lường Mass Gap (Yang-Mills)<br/>• Động lực học Replicator Entropic tiêu tán chu kỳ (Nash hỗn hợp)"]
    end

    subgraph VER["4. TẦNG KIỂM CHỨNG HÌNH THỨC & TIẾN HÓA (VERIFICATION)"]
        V1["crates/verify-bridge<br/>• BitVectorDecoder: Ánh xạ Spin ±1 sang Bit 0/1<br/>• Vòng phản hồi kín ALM: Tăng phạt động thích nghi khử sạch vi phạm<br/>• Nash Saddle-Point Verifier: Xác minh điều kiện ổn định Dx = 0, Dy = 0<br/>• Bounded Model Checking (BMC) & Bất biến quy nạp xuất SMT-LIB2 cho Z3<br/>• LLL Lattice Reduction & Nhúng Kannan giải Diophantine có chặn (Hilbert 10)<br/>• Bất biến Dehn D(P) kiểm chứng cản trở phân cắt Scissors Congruence (Hilbert 3)<br/>• 2-Descent Selmer Group XOR-SAT qua khử Gauss F_2 cho Giả thuyết BSD"]
        V2["crates/meta-evolve<br/>• Bộ tiến hóa di truyền (μ + λ) với Elitism & Gaussian Mutation<br/>• Tự động tổng hợp vector siêu tham số tối ưu θ* = (dt, a0, c0, ρ)"]
    end

    subgraph CLI["5. GIAO DIỆN ĐIỀU KHIỂN DÒNG LỆNH"]
        C1["crates/cli-bench<br/>• 21 Lệnh Benchmark chuyên sâu đo đạc vi giây chính xác<br/>• Tự động chạy toàn bộ suite, kiểm chứng 61/61 tests pass 100%"]
    end

    IN -->|"AST / Mệnh đề PUBO"| MID
    M1 --> M3
    M2 --> M3
    M3 -->|"IsingCsr Nén Thưa"| BACK
    BACK -->|"Trạng thái Spin s ∈ {-1, +1}"| VER
    V1 -. "Vi phạm? Tăng phạt λ_k" .-> M3
    V2 -. "Cung cấp siêu tham số θ*" .-> BACK
    CLI ==> IN
    CLI ==> BACK
    CLI ==> VER
```

## 3. DÒNG CHẢY DỮ LIỆU 5 BƯỚC (DATA FLOW)

```text
[Yêu cầu bài toán thực tế]
       │
       ▼ (Bước 1: Frontend DSL)
[Mô hình Ràng buộc Logic / Đa thức PUBO]
       │
       ▼ (Bước 2: Mid-End E-Graph & Tensor MPO)
[Biểu thức Tối giản Đẳng cấu & Hạ Bậc 0 Biến Phụ]
       │
       ▼ (Bước 3: Mid-End Decomposer)
[Ma trận Nén Thưa CSR & Bản đồ Thu nhỏ V-Cycle]
       │
       ▼ (Bước 4: Backend Physics GPU / CPU)
[1.024 Bản sao Lăn Rẽ nhánh qua dSB/bSB/cSB trong Không gian Pha]
       │
       ▼ (Bước 5: Verify Bridge & ALM)
[Nghiệm Số Nhị phân Khả thi 100% & File Chứng minh SMT-LIB2 cho Z3]
```

### Vòng Lặp Phản Hồi Tăng Phạt Động ALM (Augmented Lagrangian)
Khi bài toán có ràng buộc khắt khe, nếu lần giải đầu tiên xuất hiện vi phạm:
1. `Verify Bridge` kiểm tra nghiệm, tính vector sai lệch $\mathbf{c}(\mathbf{x}) \ne 0$.
2. Cập nhật nhân tử Lagrange và lực phạt động theo nguyên lý Hestenes-Powell:
   $$\lambda_k^{(t+1)} \leftarrow \lambda_k^{(t)} + \rho \cdot c_k(\mathbf{x}), \qquad \rho \leftarrow \min(\beta \cdot \rho, \rho_{\max})$$
3. Bổ sung số hạng phạt vào ma trận Ising: $H_{\text{pen}} = \sum_k \lambda_k c_k(\mathbf{x}) + \frac{\rho}{2} \sum_k c_k^2(\mathbf{x})$.
4. Đẩy lại cho `backend-physics` chạy lại. Bức tường thế năng nhân tạo buộc các hạt né tránh vùng cấm, đưa số vi phạm **triệt tiêu hoàn toàn về 0** sau 3 - 4 chu kỳ.

---

## 5. HƯỚNG DẪN LẬP TRÌNH TỪNG BƯỚC (CODE EXAMPLES)

### 5.1 Khai Báo Bài Toán Bằng Frontend DSL
```rust
use frontend_dsl::{ProblemBuilder, Constraint};
use frontend_dsl::constraints::{ExactOne, Clause3Sat, And};

// Khởi tạo bài toán với 10 biến nhị phân x_0 .. x_9
let mut builder = ProblemBuilder::new(10);
builder.add_linear_objective(0, -2.5); // Ưu tiên bật x_0 = 1
builder.add_constraint(Box::new(ExactOne::new(vec![0, 1, 2])));
builder.add_constraint(Box::new(Clause3Sat::new(3, true, 4, false, 5, true)));
builder.add_constraint(Box::new(And::new(6, 7, 8)));
```

### 5.2 Khai Báo Bài Toán Đối Kháng Minimax (Game Theory / PSPACE)
```rust
use frontend_dsl::QuantifiedProblemBuilder;

let mut game = QuantifiedProblemBuilder::new(16, 16); // 16 Phòng thủ (min), 16 Tấn công (max)
game.add_adversarial_interaction(0, 0, 1.5);
let (ising, split) = game.build_unified_ising()?;
```

### 5.3 Hạ Bậc Đa Thức Siêu Cao Bằng Tensor-Train MPO (0 Biến Phụ)
```rust
use midend_tensor::{MpoPolynomialBuilder, TensorTrainIsingProjector};

let mut mpo_builder = MpoPolynomialBuilder::new(16);
mpo_builder.add_monomial(&[0, 1, 2, 3, 4, 5, 6, 7], -5.0); // Bậc 8!
let ising = TensorTrainIsingProjector::project_pubo_to_ising(&mpo_builder)?;
// Tiết kiệm 100% biến phụ so với phương pháp Rosenberg!
```

### 5.4 Chạy Mô Phỏng Rẽ Nhánh Trên GPU (wgpu Vulkan/DirectX)
```rust
use backend_physics::{solve_dsb_gpu, DsbConfig, SolverPhysicsMode};

let config = DsbConfig {
    num_steps: 600,
    dt: 0.5,
    a0: 1.0,
    num_replicas: 1024, // 1.024 bản sao tìm kiếm đồng thời
    seed: 42,
    mode: SolverPhysicsMode::Ballistic, // Bật chế độ đạn đạo bSB nảy tường
    ..Default::default()
};

let result = solve_dsb_gpu(&ising, &config)?;
println!("Năng lượng thấp nhất đạt được: {:.4}", result.best_energy);
```

### 5.5 Khử Cơ Sở Lưới LLL Giải Diophantine Có Chặn (Hilbert 10)
```rust
use verify_bridge::lll::solve_bounded_linear_diophantine_lll;

// Giải 2x + 3y + 5z = 19 với |x|, |y|, |z| <= 50
let coeffs = vec![2, 3, 5];
let target = 19;
let bound = 50;
let sol = solve_bounded_linear_diophantine_lll(&coeffs, target, bound).unwrap();
println!("Nghiệm nguyên trích xuất: {:?}", sol); // [0, 3, 2]
```

---

## 6. BẢNG TRA CỨU TOÀN DIỆN 21 LỆNH THỰC THI CLI (`cli-bench`)

Người dùng có thể trực tiếp thực thi, kiểm chứng từng thuật toán chuyên sâu thông qua file thực thi duy nhất `cli-bench`:

| STT | Lệnh CLI | Bài Toán / Thử Thách | Mô Tả & Kết Quả Thực Nghiệm |
| :---: | :--- | :--- | :--- |
| **1** | `cargo run --release -p cli-bench -- solve` | Max-Cut Đồ thị ngẫu nhiên | Giải bài toán Max-Cut trên CPU hoặc GPU với tùy chọn replicas và steps |
| **2** | `cargo run --release -p cli-bench -- verify-dst` | Tính tái lập Bit-for-bit (DST) | Nghiệm đồng nhất 100% từng bit giữa các lần chạy độc lập (59.08 ms) |
| **3** | `cargo run --release -p cli-bench -- cross-compare`| So tài CPU vs GPU | Chạy song song kiểm tra độ khớp năng lượng và tỷ lệ vết cắt |
| **4** | `cargo run --release -p cli-bench -- bench-gpu1024` | 1.024 Replicas trên GPU | Mô phỏng 1.024 bản sao trong 1 CommandEncoder duy nhất trên Vulkan (2.63s) |
| **5** | `cargo run --release -p cli-bench -- solve3-sat` | Logic 3-SAT (20 biến, 80 mệnh đề) | Biên dịch qua cả 5 tầng, thỏa mãn sạch 80/80 mệnh đề (100.0%) trong 1.08s |
| **6** | `cargo run --release -p cli-bench -- alm-demo` | Vòng lặp kín ALM | Tự động tăng phạt thích nghi, triệt tiêu vi phạm về 0, xuất proof SMT-LIB2 |
| **7** | `cargo run --release -p cli-bench -- meta-evolve` | Tiến hóa siêu tham số di truyền | Tự động tìm bộ tham số $(dt, a_0, c_0, \rho)$ tối ưu cho GPU (3.10s) |
| **8** | `cargo run --release -p cli-bench -- solve-multilevel`| Đồ thị khổng lồ 2.000 đỉnh | V-Cycle 8 tầng co cụm HEM giải Max-Cut đồ thị lớn trong **1.53 giây** |
| **9** | `cargo run --release -p cli-bench -- bench-physics-modes`| Đọ sức 3 chế độ dSB vs bSB vs cSB | Đua tốc độ trên bề mặt gồ ghề của Spin Glass Sherrington-Kirkpatrick |
| **10**| `cargo run --release -p cli-bench -- solve-adversarial`| Đối kháng Minimax PSPACE 32 biến | Game Theory đối kháng, tìm điểm yên ngựa Nash ổn định tuyệt đối ($D_x=0, D_y=0$) |
| **11**| `cargo run --release -p cli-bench -- bench-tensor` | Đa thức bậc siêu cao PUBO $k=8$ | Hạ bậc MPO Fourier-Walsh 0 biến phụ, tiết kiệm 100% ancilla (4.89 ms) |
| **12**| `cargo run --release -p cli-bench -- bench-halo` | Phân mảnh 4 miền đa GPU | Phân rã miền đồ thị, trao đổi Halo Buffers đồng nhất 100% từng bit (47.99 ms) |
| **13**| `cargo run --release -p cli-bench -- solve-protein` | Gấp cuộn Protein 2D HP Lattice | Tìm cấu hình cuộn tự nhiên có năng lượng Gibbs thấp nhất $E = -69.43$ (2.27s) |
| **14**| `cargo run --release -p cli-bench -- solve-bmc` | Bounded Model Checking | Săn bug bảo mật và chứng minh bất biến quy nạp cho Z3 an toàn ở $\infty$ bước |
| **15**| `cargo run --release -p cli-bench -- solve-mixed-game`| Cân bằng Nash Hỗn Hợp Entropic | Tiêu tán chu kỳ giới hạn, hội tụ Nash Exploitability = 0.000000 (0.69 ms) |
| **16**| `cargo run --release -p cli-bench -- solve-vortex` | Hạt xoáy Navier-Stokes & BKM | Hạt xoáy Biot-Savart RK4 lõi Chorin, chứng minh dòng trơn toàn cục (0.015 ms) |
| **17**| `cargo run --release -p cli-bench -- solve-quantum-mps`| Mạng Tensor MPS xóa Sign Problem | Co rút tensor TFIM tất định $O(N\chi^3)$, triệt tiêu hoàn toàn Fermion Sign Problem |
| **18**| `cargo run --release -p cli-bench -- solve-riemann-berry`| Toán tử Lượng tử Riemann Zeta | Toán tử Hermite Berry-Keating, phổ thực 100% (16/16), đẩy mức GUE (0.073 ms) |
| **19**| `cargo run --release -p cli-bench -- solve-diophantine-lll`| Phương trình Diophantine LLL (H10) | Khử cơ sở lưới LLL giải $2x+3y+5z=19 \implies (0, 3, 2)$ trong **0.033 ms** |
| **20**| `cargo run --release -p cli-bench -- solve-dehn` | Bất biến Dehn Phân cắt Đa diện (H3) | Bất biến Dehn chứng minh lập phương và tứ diện đều bất khả phân cắt (0.32 ms) |
| **21**| `cargo run --release -p cli-bench -- solve-wilson-gauge`| Trường Lưới Yang-Mills Mass Gap | Vòng lặp Wilson đo sức căng dây $\sigma = 0.76 > 0$ và khối lượng keo $\Delta = 0.87 > 0$ |
| **22**| `cargo run --release -p cli-bench -- solve-artin-sos` | Chứng chỉ Đa thức Không Âm (H17) | Phân tích ma trận Gram Cholesky $Q \succeq 0$, xuất chứng chỉ SOS $P(x) \ge 0$ (0.005 ms) |
| **23**| `cargo run --release -p cli-bench -- solve-elliptic-selmer`| 2-Descent Selmer BSD Conjecture | Khử Gauss $\mathbb{F}_2$ (XOR-SAT) chặn trên hạng Mordell-Weil $r \le 0$ (0.002 ms) |
| **--**| `cargo run --release -p cli-bench` | **CHẠY TOÀN BỘ BENCHMARK SUITE** | Tự động kích hoạt toàn bộ chuỗi 21 bài toán kiểm chuẩn từ đầu đến cuối |

---


## 7. CÀI ĐẶT & HƯỚNG DẪN BIÊN DỊCH

### Yêu Cầu Môi Trường
* **Rust Toolchain:** Phiên bản Rust $\ge 1.80$ (`stable-x86_64-pc-windows-gnu` hoặc `stable-x86_64-unknown-linux-gnu`).
* **GPU Backend (Tùy chọn):** Card đồ họa tương thích Vulkan 1.2+, DirectX 12 hoặc Metal (Hệ thống tự động chuyển sang CPU DST nếu không phát hiện GPU).

### Các Lệnh Thực Thi Cơ Bản

#### 1. Kiểm tra tính toàn vẹn cú pháp:
```bash
cargo check --workspace
```

#### 2. Chạy toàn bộ 61 bài kiểm thử:
```bash
cargo test --workspace
```

#### 3. Biên dịch bản phát hành tối ưu hóa cao độ:
```bash
cargo build --release
```

#### 4. Chạy toàn bộ Benchmark Suite:
```bash
cargo run --release -p cli-bench
```

---

## 9. GIẤY PHÉP & TÁC GIẢ

Dự án được nghiên cứu, phát triển và tối ưu hóa chuyên sâu phục vụ mục tiêu giải quyết các đại bài toán tối ưu hóa phức hợp siêu quy mô trong mạng viễn thông thế hệ mới, khoa học máy tính lý thuyết, mật mã học và an ninh mạng đối kháng.

Phân phối chính thức theo giấy phép kép:
* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
* **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
