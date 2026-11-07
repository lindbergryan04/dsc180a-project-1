# SAT3 TGLF Parameter Scan & Findings Summary

## 1. Dataset Architecture
The SAT3 starter dataset consists of 43 simulation runs under `data/<case>/tglf_sat2/`. Each case has identical data structures:
- `input.tglf.gen`: 180 total plasma and numerical settings.
- `out.tglf.gbflux`: 8 values ($4 \times NS$ for $NS=2$) representing Particle Flux $\Gamma$, Heat Flux $Q$, Exchange (skipped), and Toroidal Momentum Flux $\Pi$.

### Standard Reference Baseline (`a`)
- Temperature gradients: `RLTS_1 = RLTS_2 = 3.0`
- Density gradients: `RLNS_1 = RLNS_2 = 1.0`
- Temperature ratio: `TAUS_2 = 1.0` ($T_i = T_e$)
- Collision frequency: `XNUE = 0.1`
- Geometry: `RMIN_LOC = 0.5`, `DRMAJDX_LOC = 0.0`, `KAPPA_LOC = 1.0` (circular)
- Magnetic safety factor: `Q_LOC = 2.0`, `Q_PRIME_LOC = 16.0`

## 2. Parameter Variation Map
- **Triplets (24 runs: H, D, T)**:
  - `a`: Reference baseline
  - `b, c, d`: Ion & electron temperature gradient scan (`RLTS` = 1.5, 2.25, 3.5)
  - `e, f`: Density gradient scan (`RLNS` = 2.0, 3.0)
  - `j`: Low electron collision frequency (`XNUE = 0.01`)
  - `x`: High density gradient (`RLNS = 3.0`) + low collisions (`XNUE = 0.01`)
- **Doublet (2 runs: H, T)**:
  - `y`: High density gradient (`RLNS = 3.0`) + intermediate collisions (`XNUE = 0.05`)
- **Deuterium-Only Scans (17 runs: D only)**:
  - `g, h, i`: Magnetic shear (`Q_PRIME_LOC` = 4.0, 8.0, 24.0)
  - `k`: High collision frequency (`XNUE = 1.0`)
  - `l, m`: Temperature ratio (`TAUS_2` = 0.5, 1.5)
  - `n, o`: Safety factor (`Q_LOC` = 1.5, 2.5)
  - `p, q, r`: Plasma elongation (`KAPPA_LOC` = 1.25, 1.5, 2.0)
  - `s, t, u`: Shafranov shift derivative (`DRMAJDX_LOC` = -0.125, -0.25, -0.5)
  - `v, w`: Minor radius radial location (`RMIN_LOC` = 0.75, 0.25)
  - `z`: High density gradient (`RLNS = 3.0`) + high collisions (`XNUE = 1.0`)

## 3. Deuterium Focused Scans

### A. Magnetic Shear (`Q_PRIME_LOC`)
- `g_D` (4.0): $Q_i = 18.47$, $Q_e = 7.04$, $\Pi_i = -2.26$
- `h_D` (8.0): $Q_i = 30.01$, $Q_e = 10.97$, $\Pi_i = -4.29$
- `a_D` (16.0): $Q_i = 43.10$, $Q_e = 15.26$, $\Pi_i = -10.55$
- `i_D` (24.0): $Q_i = 51.56$, $Q_e = 17.41$, $\Pi_i = -18.14$
- **Trend**: Both heat and momentum transport increase monotonically with higher magnetic shear.

### B. Collision Frequency (`XNUE`)
- Baseline drive (`RLNS = 1.0`):
  - `j_D` (0.01): $Q_i = 61.86$, $Q_e = 19.94$, $\Gamma_e = -0.093$ (particle pinch)
  - `a_D` (0.10): $Q_i = 43.10$, $Q_e = 15.26$, $\Gamma_e = 1.844$
  - `k_D` (1.00): $Q_i = 24.14$, $Q_e = 8.95$, $\Gamma_e = 1.681$
- Steep density drive (`RLNS = 3.0`):
  - `x_D` (0.01): $Q_i = 104.43$, $Q_e = 105.32$, $\Gamma_e = 37.79$
  - `f_D` (0.10): $Q_i = 40.77$, $Q_e = 43.74$, $\Gamma_e = 15.08$
  - `z_D` (1.00): $Q_i = 6.49$, $Q_e = 7.63$, $\Gamma_e = 2.45$
- **Trend**: Collisional detrapping strongly stabilizes trapped electron modes (TEM), reducing fluxes by up to 16x at high density gradients.

### C. Plasma Elongation (`KAPPA_LOC`)
- `a_D` (1.00): $Q_i = 43.10$, $Q_e = 15.26$, $\Gamma_e = 1.84$
- `p_D` (1.25): $Q_i = 47.12$, $Q_e = 17.01$, $\Gamma_e = 2.07$
- `q_D` (1.50): $Q_i = 52.00$, $Q_e = 19.37$, $\Gamma_e = 2.34$
- `r_D` (2.00): $Q_i = 66.60$, $Q_e = 25.56$, $\Gamma_e = 3.21$
- **Trend**: In normalized gyro-Bohm units, higher elongation increases transport uniformly across all channels.

## 4. Collision Frequency Isotope Comparison (H, D, T)

### Baseline Density (`RLNS = 1.0`)
- `XNUE = 0.01` (`j`):
  - H: $Q_i = 45.04$, $Q_e = 14.88$, $\Gamma_e = -0.391$
  - D: $Q_i = 61.86$, $Q_e = 19.94$, $\Gamma_e = -0.093$
  - T: $Q_i = 73.49$, $Q_e = 23.60$, $\Gamma_e = +0.214$
- `XNUE = 0.10` (`a`):
  - H: $Q_i = 34.67$, $Q_e = 12.62$, $\Gamma_e = +1.064$
  - D: $Q_i = 43.10$, $Q_e = 15.26$, $\Gamma_e = +1.844$
  - T: $Q_i = 48.73$, $Q_e = 16.74$, $\Gamma_e = +2.401$

### Steep Density (`RLNS = 3.0`)
- `XNUE = 0.01` (`x`): H: $Q_i = 78.48$, D: $Q_i = 104.43$, T: $Q_i = 122.78$ ($Q_e/Q_i \approx 1.01$)
- `XNUE = 0.05` (`y`): H: $Q_i = 48.99$, T: $Q_i = 65.18$ ($Q_e/Q_i \approx 1.05$)
- `XNUE = 0.10` (`f`): H: $Q_i = 35.83$, D: $Q_i = 40.77$, T: $Q_i = 42.97$ ($Q_e/Q_i \approx 1.08$)
- `XNUE = 1.00` (`z`): D: $Q_i = 6.49$ ($Q_e/Q_i = 1.18$)
