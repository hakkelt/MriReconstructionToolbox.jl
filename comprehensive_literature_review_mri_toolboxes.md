# Comprehensive Literature Review: MRI Reconstruction Toolboxes & Missing Algorithmic Paradigms

## Executive Summary

**MriReconstructionToolbox.jl (MRT)** is a modern, modular Julia package designed for regularized MRI reconstruction via proximal optimization algorithms (`ISTA`, `FISTA`, `ADMM`, `CG`, `CGNR`), forward operator algebra (`AbstractOperators.jl`), and flexible multi-variable decompositions (`Component`, $X = \sum_i C_i$ such as $L+S$).

While MRT offers strong variational regularization tools (Wavelets, Total Variation, Locally Low-Rank, Multi-Scale Low-Rank, Total Generalized Variation, Temporal Fourier/TV, Plug-and-Play), it currently lacks several fundamental **k-space interpolation methods**, **autocalibration algorithms**, **pre-processing/channel reduction pipelines**, **physical artifact correction operators (off-resonance, gradient non-linearity)**, **motion correction models**, **subspace/temporal-basis reconstruction**, and **quantitative parameter mapping routines** standard in established toolboxes like **BART**, **SigPy**, **Gadgetron**, **MRIReco.jl**, and **MIRT.jl**.

This document presents a review of the state of computational MRI reconstruction, provides a feature matrix of existing open-source toolboxes, reviews the foundational literature for missing methods, and outlines an architectural roadmap for MRT — including a concrete redesign of the `reconstruct` entry point around an explicit `AbstractReconstructionMethod`.

---

## 1. Feature Set Matrix of Major MRI Reconstruction Toolboxes

| Category / Feature | BART (C/Python) | SigPy (Python/C) | Gadgetron (C++/Python) | MRIReco.jl (Julia) | MIRT.jl (Julia) | MRT (Julia) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Language & Backend** | C / CUDA | Python / CuPy | C++ / CUDA | Julia / Multi-threaded | Julia / OpenCL/CUDA | Julia / Multi-threaded |
| **SENSE (Iterative)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Direct SENSE / g-factor** | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **GRAPPA (k-Space)** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **SPIRiT (Iterative k-space)**| ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **ESPIRiT (Autocalibration)**| ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **JSENSE / NLINV (Joint)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **SAKE / Structured Low-Rank**| ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Coil Whitening** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Coil Compression (SVD/GCC)**| ✅ | ✅ (scripted) | ✅ | ❌ | ❌ | ❌ |
| **Non-Cartesian (nuFFT/NFFT)**| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Density Compensation (DCF)**| ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Off-Resonance ($B_0$) Recon** | ⚠️ (via `moba`) | ❌ | ❌ | ✅ (Time-seg) | ✅ (MFI / Toeplitz) | ❌ |
| **Partial Fourier (Homodyne/POCS)**| ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Gradient Delay / RING** | ✅ (`estdelay`) | ❌ | ✅ | ❌ (ext. GIRFReco) | ❌ | ❌ |
| **Total Variation (2D/3D/T)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Wavelets (2D/3D)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Low-Rank / LLR / MSLR** | ✅ | ✅ | ❌ | ✅ | ⚠️ (unverified) | ✅ |
| **Image Decomposition ($L+S$)**| ✅ | ✅ | ❌ | ❌ | ❌ | ✅ (`Component`) |
| **Total Generalized Variation**| ✅ (`-R TGV`, `-R ICTGV`) | ❌ | ❌ | ❌ | ❌ | ✅ (`TGV2D`) |
| **Subspace / Temporal Basis** | ✅ (`pics -B`) | ✅ | ❌ | ✅ | ⚠️ (unverified) | ❌ |
| **Unrolled / Learned Recon** | ✅ (TF graphs) | ✅ (`sigpy.learn`) | ✅ (Python plugins) | ❌ | ❌ | ⚠️ (`PlugAndPlay` only) |
| **SMS / Slice-GRAPPA** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **PROPELLER / Motion Binning**| ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Parameter Mapping ($T_1, T_2$)**| ✅ (`moba`) | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Streaming / Clinical Pipeline**| ❌ | ❌ | ✅ (ISMRMRD) | ❌ | ❌ | ❌ |

**Notes on the matrix**

* ⚠️ **(unverified)** marks cells that could not be confirmed against the upstream source at the time of writing and must be checked before this table is published anywhere.
* **BART TGV:** BART's `pics` supports TGV and infimal-convolution TGV directly (`-R TGV`, `-R ICTGV`, parameterized by `alpha1:alpha0` and `gamma1:gamma2`). MRT's `TotalGeneralizedVariation2D` is therefore *not* a unique capability — the distinguishing MRT feature is that TGV composes with the same `Component`/decomposition machinery as every other regularizer.
* **MRT DCF:** as built (Stage 5), MRT provides `density_compensation(acq; method = PipeMenonDCF() | VoronoiDCF())` for **non-Cartesian** acquisitions, returning a `NonCartesianAcquisitionInfo` with `dcf` populated. Cartesian acquisitions reject it with an informative `ArgumentError` — uniform Cartesian sampling needs no density compensation, deliberately not implemented. (Earlier drafts said MRT had none at all.)
* **MRT learned recon:** `PlugAndPlay` covers denoiser-as-prox, but not unrolled networks with learned data-consistency steps (MoDL, VarNet). The distinction matters; see §2.7.2.

---

## 2. Literature Review: Missing Algorithmic Paradigms

### Category 1: Parallel Imaging, k-Space Interpolation & Autocalibration

#### 1. GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition)
* **Seminal Paper:** Griswold, M. A., et al. (2002). *Generalized autocalibrating partially parallel acquisitions (GRAPPA).* Magnetic Resonance in Medicine, 47(6), 1202–1210. [DOI: 10.1002/mrm.10171](https://doi.org/10.1002/mrm.10171).
* **Concept:** Instead of unaliasing in the image domain using sensitivity maps (like SENSE), GRAPPA estimates missing k-space points as linear combinations of neighboring acquired lines across all receiver coils. The convolution kernel weights are fitted via least squares from a fully sampled central Auto-Calibration Signal (ACS) region.
* **Constraints:** Requires uniform Cartesian undersampling with a fixed acceleration factor along the phase-encoding direction(s), plus an ACS region. It is not applicable to arbitrary or non-Cartesian sampling — this constraint must be checkable *before* reconstruction starts (see §5.9).
* **Missing in MRT:** MRT operates exclusively via image-domain proximal minimization. Adding GRAPPA requires a k-space kernel estimation and convolution routine.

#### 2. SPIRiT (Iterative Self-Consistent Parallel Imaging)
* **Seminal Paper:** Lustig, M., & Pauly, J. M. (2010). *SPIRiT: Iterative self-consistent parallel imaging reconstruction from arbitrary k-space.* Magnetic Resonance in Medicine, 64(2), 457–471. [DOI: 10.1002/mrm.22428](https://doi.org/10.1002/mrm.22428).
* **Concept:** Formulates k-space interpolation as an inverse problem enforcing consistency with convolution kernels across all coils ($k = G k$) combined with data fidelity ($\mathcal{D} k = y$) and regularizers (L1-Wavelets/TV). It bridges GRAPPA and compressed sensing for arbitrary Cartesian and non-Cartesian trajectories.
* **Missing in MRT:** No k-space convolution operator or self-consistency model.

#### 3. ESPIRiT (Eigenvalue Approach to Autocalibrating Parallel MRI)
* **Seminal Paper:** Uecker, M., et al. (2014). *ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA.* Magnetic Resonance in Medicine, 71(3), 990–1001. [DOI: 10.1002/mrm.24751](https://doi.org/10.1002/mrm.24751).
* **Concept:** Computes local calibration matrices in k-space from ACS data, constructs a block-Hankel matrix, and performs an SVD to yield null-space/signal-space kernels. Transforming signal-space kernels to the image domain produces an eigenvalue problem at each pixel: eigenvectors with eigenvalue $\approx 1$ represent smooth, high-fidelity sensitivity maps, with multi-map support resolving FOV-aliasing and phase wraps.
* **Missing in MRT:** MRT currently requires users to supply pre-computed sensitivity maps or uses analytical birdcage simulations (`coil_sensitivities`). An autocalibration module is absent.

#### 4. JSENSE & NLINV (Nonlinear Joint Estimation of Image and Sensitivity Maps)
* **Seminal Papers:**
  - Ying, L., & Sheng, J. (2007). *Joint image reconstruction and sensitivity estimation in SENSE (JSENSE).* Magnetic Resonance in Medicine, 57(6), 1196–1202. [DOI: 10.1002/mrm.21245](https://doi.org/10.1002/mrm.21245).
  - Uecker, M., Hohage, T., Block, K. T., & Frahm, J. (2008). *Nonlinear inverse reconstruction for parallel MRI.* Magnetic Resonance in Medicine, 60(3), 674–682. [DOI: 10.1002/mrm.21692](https://doi.org/10.1002/mrm.21692).
* **Concept:** Replaces sequential sensitivity calibration with joint non-linear optimization: $\min_{m, c} \| \mathcal{F}(c \cdot m) - y \|_2^2 + \alpha R_m(m) + \beta R_c(c)$, typically solved via alternating minimization or the iteratively regularized Gauss-Newton method (IRGNM).
* **Missing in MRT:** MRT models assume linear forward operators with fixed sensitivity maps. Bilinear/non-linear solvers are not yet abstracted, and `StructuredOptimization.jl` terms are built against a linear operator algebra.

#### 5. SAKE & LORAKS (Calibrationless Structured Low-Rank Matrix Completion)
* **Seminal Papers:**
  - Shin, P. J., et al. (2014). *Calibrationless parallel imaging reconstruction based on structured low-rank matrix completion (SAKE).* Magnetic Resonance in Medicine, 72(4), 959–970. [DOI: 10.1002/mrm.24997](https://doi.org/10.1002/mrm.24997).
  - Haldar, J. P. (2014). *Low-rank modeling of local k-space neighborhoods (LORAKS) for constrained MRI.* IEEE Transactions on Medical Imaging, 33(3), 668–681. [DOI: 10.1109/TMI.2013.2293974](https://doi.org/10.1109/TMI.2013.2293974).
  - Jin, K. H., et al. (2016). *A general framework for compressed sensing and parallel imaging using annihilating filter based low-rank Hankel matrix approach (ALOHA).* IEEE Transactions on Image Processing, 25(11), 5363–5376. [DOI: 10.1109/TIP.2016.2601243](https://doi.org/10.1109/TIP.2016.2601243).
* **Concept:** Exploits the property that local k-space patches across multi-channel arrays form a rank-deficient block-Hankel/Toeplitz matrix. Calibrationless parallel imaging recovers undersampled k-space directly, without explicit sensitivity calibration.
* **Formulation caveat (important for §5.5):** SAKE as published is *not* nuclear-norm minimization. It is a **Cadzow-type alternating projection**: project onto the fixed-rank set (hard truncation of the SVD to rank $r$), then project onto the data-consistency set. LORAKS likewise uses a rank-$r$ constraint or a rank penalty $J_r$, not $\min \operatorname{rank}$. Both are therefore **non-convex**: any splitting algorithm applied to them (Douglas–Rachford, ADMM) is a heuristic with no convergence guarantee, and this must be documented rather than implied away by reusing the convex-solver API. A convex nuclear-norm relaxation is a legitimate *alternative* formulation, but it is a different algorithm and should be named differently.
* **Missing in MRT:** Hankel lifting operators and structured matrix completion projections.

---

### Category 2: Pre-Processing, Noise Decorrelation & Channel Compression

#### 1. Noise Pre-Whitening & SNR Normalization
* **Seminal Paper:** Kellman, P., & McVeigh, E. R. (2005). *Image reconstruction in SNR units: A general method for SNR measurement.* Magnetic Resonance in Medicine, 54(6), 1439–1447. [DOI: 10.1002/mrm.20713](https://doi.org/10.1002/mrm.20713).
* **Concept:** Multi-channel coil arrays suffer from inductive coupling, resulting in correlated noise across channels. Acquiring a noise-only prescan measures the noise covariance matrix $\mathbf{\Psi} \in \mathbb{C}^{N_c \times N_c}$. Pre-multiplying data by $\mathbf{L}^{-1}$ (where $\mathbf{\Psi} = \mathbf{L}\mathbf{L}^H$ is the Cholesky factorization) whitens the noise, so that $\operatorname{Cov}(\mathbf{L}^{-1} n) = \mathbf{I}$. The data fidelity term then becomes a standard Euclidean distance and the least-squares solution is the maximum-likelihood estimate.
* **Missing in MRT:** No noise covariance estimation or pre-whitening utility in `acquisition_data/`.

#### 2. Software Coil Compression (SVD & Geometric Coil Compression)
* **Seminal Papers:**
  - Buehrer, M., Pruessmann, K. P., Boesiger, P., & Kozerke, S. (2007). *Array compression for MRI with large coil arrays.* Magnetic Resonance in Medicine, 57(6), 1131–1139. [DOI: 10.1002/mrm.21237](https://doi.org/10.1002/mrm.21237).
  - Huang, F., Vijayakumar, S., Li, Y., Hertel, S., & Duensing, G. R. (2008). *A software channel compression technique for faster reconstruction with many channels.* Magnetic Resonance Imaging, 26(1), 133–141. [DOI: 10.1016/j.mri.2007.04.010](https://doi.org/10.1016/j.mri.2007.04.010).
  - Zhang, T., Pauly, J. M., Vasanawala, S. S., & Lustig, M. (2013). *Coil compression for accelerated imaging with Cartesian and non-Cartesian sampling.* Magnetic Resonance in Medicine, 69(2), 571–582. [DOI: 10.1002/mrm.24267](https://doi.org/10.1002/mrm.24267).
* **Concept:** Modern receiver arrays (32–128 coils) introduce heavy computational and memory footprints. SVD-based virtual coil compression linearly combines channels into a reduced set (e.g. 8–12 virtual coils). **Geometric Coil Compression (GCC)** performs alignment along a fully sampled spatial readout dimension via local SVD, preserving $>99\%$ SNR while speeding up iterative reconstruction by 3–10×.
* **Missing in MRT:** Pre-processing transforms to compress `AcquisitionInfo.kspace_data` and the corresponding sensitivity maps consistently, before operator creation.

#### 3. Density Compensation Functions (DCF) for Non-Cartesian Trajectories
* **Seminal Papers:**
  - Pipe, J. G., & Menon, P. (1999). *Sampling density compensation in MRI: Rationale and an iterative numerical solution.* Magnetic Resonance in Medicine, 41(1), 179–186. [DOI: 10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V](https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V).
  - Jackson, J. I., Meyer, C. H., Nishimura, D. G., & Macovski, A. (1991). *Selection of a convolution function for Fourier inversion using gridding.* IEEE Transactions on Medical Imaging, 10(3), 473–478. [DOI: 10.1109/42.97598](https://doi.org/10.1109/42.97598).
* **Concept:** Non-Cartesian trajectories (radial, spiral) oversample the center of k-space. Direct adjoint operations and initial estimates $\mathcal{A}^H y$ require weighting k-space by density compensation weights (Voronoi cell area or Pipe's iterative method).
* **Missing in MRT:** MRT uses raw NFFT adjoints with no integrated DCF, which degrades both the direct reconstruction and the default initial guess $x_0 = \mathcal{A}^H y$ for iterative solves.

---

### Category 3: Partial Fourier & Phase Symmetry Techniques

#### 1. Homodyne Detection & POCS (Projection Onto Convex Sets)
* **Seminal Papers:**
  - Noll, D. C., Nishimura, D. G., & Macovski, A. (1991). *Homodyne detection in magnetic resonance imaging.* IEEE Transactions on Medical Imaging, 10(2), 154–163. [DOI: 10.1109/42.79473](https://doi.org/10.1109/42.79473).
  - Haacke, E. M., Lindskog, E. D., & Lin, W. (1991). *A fast, robust, algorithm for 2D or 3D half-Fourier reconstruction.* Journal of Magnetic Resonance, 92(1), 126–145. [DOI: 10.1016/0022-2364(91)90253-P](https://doi.org/10.1016/0022-2364(91)90253-P).
* **Concept:** Exploits conjugate (Hermitian) symmetry of k-space ($S(-k) = S^*(k)$, exact only for a real-valued image) to reconstruct from asymmetric acquisitions covering $>50\%$ of k-space along one encoding direction.
  - **Homodyne:** Estimates a low-resolution phase map $\phi_0$ from the central symmetric band, applies asymmetric ramp weighting to k-space, transforms to image domain, demodulates by $e^{-i\phi_0}$ and takes the real part. Fast and non-iterative, but discards residual phase — it is not equivalent to the iterative variants and is more sensitive to rapid phase variation.
  - **POCS:** Iteratively alternates between enforcing the estimated phase in image space and data consistency on the *acquired* samples in k-space.
* **Asymmetry note:** partial Fourier is asymmetric along **one** encoding direction, and the width of the symmetric band is determined by the sampled fraction. An isotropic, user-supplied `low_freq_size` default is therefore a poor API default (see §5.6).
* **Missing in MRT:** Asymmetric filtering operators, phase map extraction, and POCS / phase-constrained reconstruction.

#### 2. Virtual Conjugate Coils (VCC-SENSE / VCC-GRAPPA)
* **Seminal Paper:** Blaimer, M., et al. (2009). *Virtual coil concept for improved parallel MRI employing conjugate symmetric signals.* Magnetic Resonance in Medicine, 61(1), 93–102. [DOI: 10.1002/mrm.21652](https://doi.org/10.1002/mrm.21652).
* **Concept:** Synthesizes virtual conjugate channels from the complex conjugate of the reversed k-space. Combined with SENSE or GRAPPA, this incorporates phase constraints to improve parallel imaging acceleration without explicit phase estimation.
* **Missing in MRT:** Operator abstraction for conjugate channels.

---

### Category 4: Physical Field Off-Resonance & Distortion Corrections

#### 1. $B_0$ Field Inhomogeneity & Off-Resonance Correction
* **Seminal Papers:**
  - Sutton, B. P., Noll, D. C., & Fessler, J. A. (2003). *Fast, iterative image reconstruction for MRI in the presence of field inhomogeneities.* IEEE Transactions on Medical Imaging, 22(2), 178–188. [DOI: 10.1109/TMI.2002.808360](https://doi.org/10.1109/TMI.2002.808360).
  - Fessler, J. A., et al. (2005). *Toeplitz-based iterative image reconstruction for MRI with correction for magnetic field inhomogeneity.* IEEE Transactions on Signal Processing, 53(9), 3393–3402. [DOI: 10.1109/TSP.2005.853152](https://doi.org/10.1109/TSP.2005.853152).
* **Concept:** Main field inhomogeneity causes spatial blurring and geometric distortion in non-Cartesian (spiral) and EPI acquisitions through a phase evolution term. The forward model is

  $$ (\mathcal{A} x)(t) = \int x(r)\, c(r)\, e^{-i 2\pi k(t) \cdot r}\, e^{-\left(R_2^*(r) \,+\, i\,\omega_0(r)\right) t} \, \mathrm{d}r, \qquad \omega_0(r) = \gamma \Delta B_0(r) = 2\pi \Delta f_0(r). $$

  Note the structure: $R_2^*$ is a **real** decay rate, and the off-resonance frequency enters as the **imaginary** part. (An earlier revision of this document had these two swapped, which would produce decay where there should be precession and vice versa.) Fast forward operators use **time-segmentation** or **multi-frequency interpolation (MFI)** with min-max or Taylor approximations to decompose the off-resonance forward model into a short series of standard NFFTs, $\mathcal{A} \approx \sum_{l} \operatorname{diag}(b_l)\, \mathcal{F}\, \operatorname{diag}(c_l)$.
* **Missing in MRT:** No off-resonance forward operator or $B_0$ map integration. This is a natural first consumer of the `signal_model` slot proposed in §5.4.

#### 2. EPI Nyquist Ghost Correction (Phase Correction)
* **Seminal Paper:** Bruder, H., Fischer, H., Reinfelder, H.-E., & Schmitt, F. (1992). *Image reconstruction for echo planar imaging with nonequidistant k-space sampling.* Magnetic Resonance in Medicine, 23(2), 311–323. [DOI: 10.1002/mrm.1910230211](https://doi.org/10.1002/mrm.1910230211).
* **Concept:** EPI alternates gradient polarities, causing 1D/2D linear phase discrepancies between even and odd echoes that produce $N/2$ Nyquist ghosting. 1D navigator lines, SVD-based unmixing, or entropy-minimization phase corrections are standard prerequisites.
* **Missing in MRT:** EPI phase-correction pre-processing.

#### 3. Gradient Delay Correction & RING (Radial Intersections)
* **Seminal Paper:** Rosenzweig, S., Holme, H. C. M., & Uecker, M. (2019). *Simple auto-calibrated gradient delay estimation from few spokes using radial intersections (RING).* Magnetic Resonance in Medicine, 81(3), 1898–1906. [DOI: 10.1002/mrm.27506](https://doi.org/10.1002/mrm.27506).
* **Concept:** In non-Cartesian imaging (particularly radial, golden-angle radial, and rosette trajectories), hardware gradient delays and eddy currents cause trajectory shifts and severe streaking/ring-like artifacts.
  - In an ideal radial acquisition, all spokes intersect at the origin ($k=0$). With anisotropic gradient delays, the intersection points of spoke pairs shift and trace an ellipse in k-space.
  - **RING** auto-calibrates the **2D** gradient delay matrix (3 independent parameters) directly from as few as 3 acquired radial spokes by fitting these intersection points, requiring no external calibration scan or field probes. The original publication is 2D; 3D application is a later extension and should not be attributed to the seminal paper.
* **Toolbox Status:**
  - **BART:** Natively implements RING via `bart estdelay -R` (and trajectory correction via `bart traj -c / -q`).
  - **Gadgetron:** Supports gradient delay estimation via cross-correlation calibration modules.
  - **MRT & MRIReco.jl:** Missing native auto-calibrated gradient delay estimation. Nominal trajectories are assumed exact.

---

### Category 5: Motion Management & Dynamic Imaging

#### 1. PROPELLER / BLADE Reconstruction
* **Seminal Paper:** Pipe, J. G. (1999). *Motion correction with PROPELLER MRI: Application to head motion and free-breathing cardiac imaging.* Magnetic Resonance in Medicine, 42(5), 963–969. [DOI: 10.1002/(SICI)1522-2594(199911)42:5<963::AID-MRM17>3.0.CO;2-L](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C963::AID-MRM17%3E3.0.CO;2-L).
* **Concept:** Collects k-space in rotating rectangular strips (blades). The central circular disc is sampled by every blade, allowing blade-to-blade estimation of 2D rigid-body rotation, translation, and phase variation. Corrupted blades are re-weighted or rejected before gridding.
* **Missing in MRT:** Inter-blade registration, phase correction, and correlation-weighted reconstruction.

#### 2. XD-GRASP (Extra-Dimensional Golden-Angle Radial Sparse Parallel MRI)
* **Seminal Paper:** Feng, L., et al. (2016). *XD-GRASP: Golden-angle radial MRI with reconstruction of extra motion-state dimensions using compressed sensing.* Magnetic Resonance in Medicine, 75(2), 775–788. [DOI: 10.1002/mrm.25665](https://doi.org/10.1002/mrm.25665).
* **Concept:** Uses continuous golden-angle radial acquisition to extract self-navigation signals (respiratory or cardiac motion curves from the k-space center) and sorts data into multi-dimensional motion-state bins (e.g. contrast × respiratory × cardiac), followed by multi-dimensional regularized reconstruction (TV across the extra dimensions).
* **Fit with MRT:** the *reconstruction* half of XD-GRASP is already expressible today — once data are binned, `TemporalTotalVariation` over the extra dimension plus the existing task-splitting machinery covers it. What is missing is the **binning half**: self-gating signal extraction, motion-state sorting, and the resulting non-uniform per-bin trajectories.
* **Missing in MRT:** Self-gating signal extraction and multi-dimensional binning utilities.

---

### Category 6: Quantitative Parameter Mapping & Water-Fat Separation

#### 1. Model-Based Parameter Mapping (Direct $T_1$, $T_2$, $T_2^*$ Estimation)
* **Seminal Papers:**
  - Block, K. T., Uecker, M., & Frahm, J. (2009). *Model-based iterative reconstruction for radial fast spin-echo MRI.* IEEE Transactions on Medical Imaging, 28(11), 1759–1769. [DOI: 10.1109/TMI.2009.2023119](https://doi.org/10.1109/TMI.2009.2023119).
  - Sumpf, T. J., et al. (2011). *Model-based nonlinear inverse reconstruction for T2 mapping using highly undersampled spin-echo MRI.* Journal of Magnetic Resonance Imaging, 34(2), 420–428. [DOI: 10.1002/jmri.22633](https://doi.org/10.1002/jmri.22633).
  - Wang, X., et al. (2018). *Model-based T1 mapping with sparsity constraints using single-shot inversion-recovery radial FLASH.* Magnetic Resonance in Medicine, 79(2), 730–740. [DOI: 10.1002/mrm.26726](https://doi.org/10.1002/mrm.26726).
* **Concept:** Instead of reconstructing a series of contrast images and subsequently fitting relaxation models pixel-by-pixel, model-based reconstruction integrates the physical signal equation (e.g. $S(t) = M_0 (1 - 2 e^{-t/T_1})$ for ideal inversion recovery, $S(t) = M_0 e^{-t/T_2}$ for spin echo) directly into a non-linear forward model, and solves for the parameter maps.
* **Missing in MRT:** Physics-based non-linear forward models for quantitative MRI. Note that MRT's existing `NonNegative` and `BoxConstraint` regularizers were designed with parameter maps in mind, so the regularization side is partly ready; the solver side is not.

#### 2. Water-Fat Separation (Dixon Techniques & IDEAL)
* **Seminal Papers:**
  - Reeder, S. B., et al. (2005). *Iterative decomposition of water and fat with echo asymmetry and least-squares estimation (IDEAL): Application with fast spin-echo imaging.* Magnetic Resonance in Medicine, 54(3), 636–644. [DOI: 10.1002/mrm.20624](https://doi.org/10.1002/mrm.20624).
  - Hernando, D., Kellman, P., Haldar, J. P., & Liang, Z.-P. (2010). *Robust water/fat separation in the presence of large field inhomogeneities using a graph cut algorithm.* Magnetic Resonance in Medicine, 63(1), 79–90. [DOI: 10.1002/mrm.22177](https://doi.org/10.1002/mrm.22177).
* **Concept:** Multi-echo gradient-echo acquisitions model the signal as a superposition of water and multi-peak fat chemical shifts subject to local field inhomogeneity $\psi$ (in Hz) and, in the full model, $R_2^*$ decay:

  $$ s(t_n) = \Big(W + F \sum_p \alpha_p e^{i 2\pi \Delta f_p t_n}\Big)\, e^{\left(i 2\pi \psi - R_2^*\right) t_n}. $$

  Graph-cut or non-linear multi-variable optimization estimates water, fat, and the field map jointly; the field map is the source of the well-known water/fat swap ambiguity.
* **Missing in MRT:** Multi-chemical species modeling and graph-cut field map estimators.

---

### Category 7: Additional Paradigms (previously unlisted)

#### 1. Subspace / Temporal-Basis Reconstruction
* **Seminal Papers:**
  - Liang, Z.-P. (2007). *Spatiotemporal imaging with partially separable functions.* IEEE ISBI 2007, 988–991. [DOI: 10.1109/ISBI.2007.357020](https://doi.org/10.1109/ISBI.2007.357020).
  - Tamir, J. I., et al. (2017). *T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging.* Magnetic Resonance in Medicine, 77(1), 180–195. [DOI: 10.1002/mrm.26102](https://doi.org/10.1002/mrm.26102).
* **Concept:** Model the image time-series as $X = \Phi \alpha$, where $\Phi \in \mathbb{C}^{N_t \times K}$ is a *pre-computed* low-dimensional temporal basis (from Bloch simulations or an SVD of a signal dictionary) and $\alpha$ holds $K \ll N_t$ spatial coefficient maps. Reconstruction solves $\min_\alpha \tfrac{1}{2}\|\mathcal{A}\Phi\alpha - y\|_2^2 + R(\alpha)$, drastically reducing the number of unknowns while remaining fully linear and convex. BART exposes this as `pics -B <basis>`.
* **Why this matters most for MRT:** unlike almost every other gap in this document, subspace reconstruction needs **no new solver and no new non-linear machinery**. It is a linear operator $\Phi$ composed into $\mathcal{A}$ — exactly the `signal_model` slot proposed in §5.4 — and it composes directly with MRT's existing `LowRank`, `LocallyLowRank`, `MultiScaleLowRank` and `Component` infrastructure, which is where MRT is already strongest. It is arguably the highest value-per-effort addition in the whole roadmap and is scheduled accordingly in §3.
* **Missing in MRT:** temporal basis operator and basis-estimation utilities.

#### 2. Unrolled / Learned Reconstruction
* **Seminal Papers:**
  - Hammernik, K., et al. (2018). *Learning a variational network for reconstruction of accelerated MRI data.* Magnetic Resonance in Medicine, 79(6), 3055–3071. [DOI: 10.1002/mrm.26977](https://doi.org/10.1002/mrm.26977).
  - Aggarwal, H. K., Mani, M. P., & Jacob, M. (2019). *MoDL: Model-based deep learning architecture for inverse problems.* IEEE Transactions on Medical Imaging, 38(2), 394–405. [DOI: 10.1109/TMI.2018.2865356](https://doi.org/10.1109/TMI.2018.2865356).
  - Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). *Plug-and-play priors for model based reconstruction.* IEEE GlobalSIP 2013, 945–948. [DOI: 10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048).
* **Concept:** Replace hand-crafted priors with learned ones. **Plug-and-Play** substitutes a denoiser for the proximal operator inside an otherwise standard splitting algorithm — MRT already supports this via `PlugAndPlay`. **Unrolled networks** (VarNet, MoDL) go further: they unroll a fixed number of gradient/data-consistency steps into a differentiable graph and train the regularizer *and* step sizes end-to-end. The latter is a fundamentally different execution model (training loop, autodiff through the forward operator) and does not fit the current `reconstruct` contract.
* **Missing in MRT:** unrolled architectures and any training infrastructure. Whether this belongs in MRT at all, or in a downstream package depending on it, is an open scoping question (§5.10).

#### 3. Simultaneous Multi-Slice (SMS) / Slice-GRAPPA
* **Seminal Papers:**
  - Setsompop, K., et al. (2012). *Blipped-controlled aliasing in parallel imaging for simultaneous multislice echo planar imaging with reduced g-factor penalty.* Magnetic Resonance in Medicine, 67(5), 1210–1224. [DOI: 10.1002/mrm.23097](https://doi.org/10.1002/mrm.23097).
  - Cauley, S. F., et al. (2014). *Interslice leakage artifact reduction technique for simultaneous multislice acquisitions.* Magnetic Resonance in Medicine, 72(1), 93–102. [DOI: 10.1002/mrm.24898](https://doi.org/10.1002/mrm.24898).
* **Concept:** Excite several slices simultaneously and separate them using coil sensitivity differences, enhanced by blipped-CAIPI inter-slice shifts. Split slice-GRAPPA additionally suppresses inter-slice signal leakage.
* **Missing in MRT:** SMS encoding model and slice-GRAPPA kernels. Notably, SMS interacts badly with MRT's automatic problem decomposition over slices (§5.8) — slices are no longer separable subproblems, and the decomposition planner would need to know this.

#### 4. g-Factor & Noise Propagation (Pseudo-Replica)
* **Seminal Papers:**
  - Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P. (1999). *SENSE: Sensitivity encoding for fast MRI.* Magnetic Resonance in Medicine, 42(5), 952–962. [DOI: 10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S).
  - Robson, P. M., et al. (2008). *Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions.* Magnetic Resonance in Medicine, 60(4), 895–907. [DOI: 10.1002/mrm.21728](https://doi.org/10.1002/mrm.21728).
* **Concept:** The g-factor quantifies spatially varying noise amplification from parallel imaging. For non-linear or iterative reconstructions no closed form exists, and the **pseudo-replica** method estimates it empirically by repeating the reconstruction with synthetic noise realizations added to the data.
* **Fit with MRT:** the pseudo-replica method is embarrassingly parallel and maps cleanly onto MRT's existing `ReconstructionExecutor` / task-splitting infrastructure — it is a loop over the same `reconstruct` call with perturbed data. Low implementation cost, high diagnostic value, and it applies to *every* method in this document.
* **Missing in MRT:** g-factor and SNR-map utilities.

---

## 3. Prioritized Implementation Roadmap for MRT

To expand MRT's utility while preserving its clean, Julia-idiomatic architecture, we recommend phased development. **Phase 0 is new and blocking**: several formulations in §5.5 depend on solvers and API surface that do not yet exist in MRT.

```
Phase 0: Enabling Infrastructure (blocking prerequisites)
├── 0.1 `AbstractReconstructionMethod` refactor of `reconstruct` (§5) — clean break, no shim (done)
├── 0.2 Solver integration: alias/export DouglasRachford (and AFBA if needed),
│        extend `patch_algorithm_with_default_values`, verify StructuredOptimization
│        supports the resulting term structures
└── 0.3 `signal_model` slot on the forward operator (composition 𝒜∘M) — consumed by
         subspace recon, B₀ correction, phase demodulation and quantitative models

Phase 1: Pre-processing & Calibration (immediate impact, no new math)
├── 1.1 Noise Pre-Whitening (noise covariance estimation & Cholesky whitening)
├── 1.2 Coil Compression (SVD & Geometric Coil Compression GCC)
├── 1.3 Density Compensation Functions (Pipe's iterative DCF, Voronoi for NFFT)
└── 1.4 ESPIRiT Calibration (eigenvalue decomposition of ACS block-Hankel matrices)

Phase 2: Highest Value-per-Effort Linear Extensions
├── 2.1 Subspace / Temporal-Basis Reconstruction (𝒜∘Φ; reuses Phase 0.3)
└── 2.2 g-Factor / Pseudo-Replica SNR maps (reuses existing executor infrastructure)

Phase 3: Classical Parallel Imaging & Partial Fourier
├── 3.1 Gradient Delay Correction & RING (auto-calibrated radial trajectory correction)
├── 3.2 Partial Fourier (Homodyne filter, POCS, phase-constrained iterative)
└── 3.3 GRAPPA / SPIRiT k-space interpolation (requires k-space domain, Phase 0.1)

Phase 4: Physical & Artifact Correction Operators
├── 4.1 Off-Resonance Forward Operator (time-segmented / MFI B₀ + R₂* correction)
└── 4.2 EPI Nyquist Ghost Phase Correction

Phase 5: Advanced Non-linear & Quantitative Reconstruction
├── 5.1 Non-linear / Bilinear Solvers (JSENSE / NLINV, IRGNM)
├── 5.2 Structured Low-Rank Matrix Completion (SAKE / LORAKS; non-convex — see §2.1.5)
└── 5.3 Model-Based Quantitative Parameter Mapping (T₁, T₂, Dixon Water-Fat)

Out of scope for now (see §5.10): unrolled/learned reconstruction, SMS, PROPELLER,
streaming pipelines.
```

**Rationale for the reordering versus the previous draft:** DCF moved up into Phase 1 because it silently degrades every existing non-Cartesian reconstruction today, including the default initial guess. Subspace reconstruction and pseudo-replica g-factor were promoted to Phase 2 because they are linear, need no new solver, and reuse infrastructure MRT already has — they deliver more per unit of effort than GRAPPA does.

---

## 4. Key Takeaways & Recommendations

1. **Leverage Julia ecosystem strengths.** MRT's operator-composition framework (`AbstractOperators.jl`) and proximal solvers (`StructuredOptimization.jl` / `ProximalAlgorithms.jl`) are well suited to variational problems, and the `Component` decomposition API is genuinely differentiating.
2. **Separate the three concepts that the current API conflates.** A reconstruction is specified by (a) a **forward/signal model** (what maps the unknown to the data: sensitivities, Fourier, phase, $B_0$, temporal basis), (b) an **objective** (data fidelity plus regularizers/constraints), and (c) a **solver**. Today all three are entangled in `reconstruct`'s positional arguments, which is precisely why GRAPPA, POCS and homodyne cannot be expressed. §5 separates them.
3. **Keep modular separation for pre-processing.** Pre-processing transforms (whitening, GCC, ESPIRiT, RING, DCF) should return updated `AcquisitionInfo` instances so they chain seamlessly ahead of the reconstruction pipeline, rather than becoming keyword arguments of `reconstruct`.
4. **Prefer types over symbols on every dispatch axis.** MRT already dispatches on `Regularization` subtypes and on tuple shape; the new API should not introduce stringly-typed configuration that defeats that (§5.3).
5. **Be explicit about convexity.** Several attractive methods (SAKE, LORAKS, rank-constrained variants) are non-convex heuristics. Reusing a convex solver's API for them is fine; silently implying convergence guarantees is not.

---

## 5. Architectural Redesign: Unified `ReconstructionMethod` Interface for `reconstruct`

### 5.1 Motivation & Design Goals

The original `reconstruct` had the positional signature `reconstruct(acq_data, [regularization], [algorithm]; kwargs...)`.
(As built: this signature was **deleted outright** — clean break, no deprecation shim.)
This was natural for proximal minimization $\min_x \tfrac{1}{2}\|\mathcal{A}x - y\|_2^2 + \sum_i R_i(x)$, but it tightly couples `reconstruct` to image-space iterative regularization. Alternative reconstruction methods (`GRAPPA`, `Homodyne`, `SAKE`, or the direct adjoint $\mathcal{A}^H y$) cannot be expressed at all.

By taking an explicit `method::AbstractReconstructionMethod` as the second positional argument:
```julia
reconstruct(acq_data::AcquisitionInfo, method::AbstractReconstructionMethod = DirectReconstruction(); kwargs...)
```
all reconstruction algorithms share a single, polymorphic, extensible dispatch layer.

**Design goals, in priority order:**

1. **Every configuration axis that selects code must be a type, not a `Symbol`** — so it dispatches, specializes, and can be extended by downstream packages.
2. **Forward-model factors, objective terms, and solvers are separate fields** — they are separate mathematical objects (§4.2).
3. **Named methods are types**, so that `SPIRiT` can be dispatched on, shown, and validated, rather than dissolving into a generic `IterativeReconstruction` at construction time.
4. **Backwards compatibility via an explicit deprecation path**, not a silent `MethodError`.

**A capability gained by this refactor, worth stating explicitly:** unregularized *iterative* least squares is currently unreachable. In `reconstruct.jl`, `regularization == ()` short-circuits to the direct adjoint, so `CG`/`CGNR` can never run without a regularizer. Under the new API this is simply `IterativeReconstruction(; algorithm = CGNR())`, cleanly distinct from `DirectReconstruction()`.

---

### 5.2 Proposed Type Hierarchy

The taxonomy below separates the *kind* of reconstruction (its axis of classification) from the *named methods*, which was conflated in the previous draft — `GRAPPA` and `Homodyne` are both direct methods, so listing them as siblings of `DirectReconstruction` implied a structure that does not hold.

```
AbstractReconstructionMethod
├── AbstractIterativeMethod
│   ├── IterativeReconstruction        # the general engine: model + objective + solver
│   └── (named presets, all lowering to IterativeReconstruction)
│       ├── SPIRiT
│       ├── SAKE
│       ├── LORAKS
│       ├── POCS
│       └── PhaseConstrained   (as built: a direct method, see §5.7)
└── AbstractDirectMethod
    ├── DirectReconstruction           # zero-filled adjoint 𝒜'y / gridding, optional DCF
    ├── GRAPPA                         # k-space convolution kernel synthesis
    └── Homodyne                       # asymmetric filtering & phase demodulation
```

```julia
"""
    AbstractReconstructionMethod

Abstract supertype for all MRI reconstruction methods in MRT.

# Interface

A concrete method must implement either a `_reconstruct_dispatch` method, or `lower` (below).

- `lower(method) -> AbstractReconstructionMethod`: rewrite a named preset into the general
  engine that executes it. Defaults to `identity`, so engines need not implement it.
- `check_applicable(method, acq_data) -> nothing`: throw an informative `ArgumentError` if the
  method cannot be applied to this acquisition (§5.9). Defaults to a no-op.
"""
abstract type AbstractReconstructionMethod end

abstract type AbstractIterativeMethod <: AbstractReconstructionMethod end
abstract type AbstractDirectMethod <: AbstractReconstructionMethod end

"""
    lower(method) -> AbstractReconstructionMethod

Rewrite a named reconstruction method into the general method that implements it, so that
`SPIRiT(...)` and the equivalent hand-written `IterativeReconstruction(...)` execute through
exactly one code path. Methods that *are* the engine return themselves.
"""
lower(method::AbstractReconstructionMethod) = method
```

---

### 5.3 Configuration Axes as Types

The previous draft used `domain::Symbol` and `data_fidelity::Symbol`, validated at construction with `@argcheck`. That defeats the stated goal of a polymorphic dispatch layer: the two fields that actually select the algorithm would have to be tested at runtime in an `if`-chain, they cannot be extended from outside the package, and because they are not type parameters the compiler cannot specialize on them. Both become singleton types instead.

```julia
"""
    ReconstructionDomain

In which space the optimization variable lives. `ImageDomain` optimizes over the image
`x ∈ ℂ^(Nx × Ny × …)`; `KSpaceDomain` optimizes over the multi-channel k-space array
`k ∈ ℂ^(Nkx × Nky × … × Nc)`.
"""
abstract type ReconstructionDomain end

struct ImageDomain <: ReconstructionDomain end

"""
    KSpaceDomain(; coil_combine = AdjointSensitivity())

Optimize over multi-channel k-space. Because `reconstruct` returns an image, a k-space-domain
method must say how the final coil images are combined; see [`CoilCombination`](@ref).
"""
Base.@kwdef struct KSpaceDomain{C} <: ReconstructionDomain
    coil_combine::C = AdjointSensitivity()
end

"""
    CoilCombination

How a `KSpaceDomain` reconstruction turns its solved k-space `k̂` into the returned image.

- `AdjointSensitivity()`: `𝒮' * ℱ⁻¹ k̂` using the acquisition's sensitivity maps. SNR-optimal
  and phase-preserving; requires `acq_data.sensitivity_maps`, and errors if absent.
- `RootSumSquares()`: `sqrt.(sum(abs2, ℱ⁻¹ k̂; dims = coil_dim))`. Calibration-free — the right
  default for genuinely calibrationless methods (SAKE, LORAKS, PRUNO) — but discards phase and
  is not SNR-optimal.
- `NoCoilCombination()`: return the multi-channel coil images without combining. Escape hatch
  for callers doing their own combination or inspecting intermediate results.
"""
abstract type CoilCombination end
struct AdjointSensitivity <: CoilCombination end
struct RootSumSquares    <: CoilCombination end
struct NoCoilCombination <: CoilCombination end

"""
    DataFidelity

How consistency with the measured data enters the objective.

- `L2Loss()`: prepend the quadratic term `½‖𝒜x − y‖₂²` (image domain) or `½‖𝒫k − y‖₂²`
  (k-space domain).
- `HardConsistency()`: prepend the indicator of the affine consistency set,
  `i_{𝒫k = y}` or `i_{𝒜x = y}`, enforced by projection (evaluated in closed form when
  `𝒜𝒜*` is diagonal, or via inner CG iterations otherwise).
- `NoFidelity()`: prepend nothing; the objective is entirely specified by `regularization`.
  Use when a data-consistency term is already present among the supplied terms.
"""
abstract type DataFidelity end
struct L2Loss          <: DataFidelity end
struct HardConsistency <: DataFidelity end
struct NoFidelity      <: DataFidelity end
```

**Notation.** $\mathcal{A}$ denotes the full image-domain encoding operator (sensitivities ∘ Fourier ∘ sampling), as built by `get_encoding_operator`. $\mathcal{P}$ denotes the **sampling and data-consistency operator** (was $\Gamma$ / $\mathcal{D}$): the restriction of a full multi-channel Cartesian k-space array to the acquired samples, i.e. the sampling operator alone, with no sensitivity or Fourier factor. $\mathcal{P}$ is the k-space-domain counterpart of $\mathcal{A}$ and is what appears in every `KSpaceDomain` formulation below. MRT's existing `get_subsampling_operator` already treats trailing dimensions as batch dimensions and carries the coil dimension through unchanged.

---

### 5.4 `IterativeReconstruction`

```julia
const DEFAULT_ALGORITHMS = (CG(), CGNR(), FISTA(), ADMM(), DouglasRachford())

"""
    IterativeReconstruction(
        regularization = ();
        algorithm = DEFAULT_ALGORITHMS,
        domain::ReconstructionDomain = ImageDomain(),
        data_fidelity::DataFidelity = L2Loss(),
        signal_model = nothing,
    )
    IterativeReconstruction(reg1, regs...; kwargs...)

Solve a regularized/variational inverse problem via proximal/gradient algorithms.

# Arguments
- `regularization`: a `Regularization`, a `Component`, or a tuple of either (but not a mix of
  both, matching the existing `reconstruct` contract).

# Keyword arguments
- `algorithm`: candidate solver(s); the applicable one is selected from the assembled model,
  as today.
- `domain`: `ImageDomain()` (default) or `KSpaceDomain(; coil_combine)` — see §5.3.
- `data_fidelity`: `L2Loss()` (default), `HardConsistency()`, or `NoFidelity()` — see §5.3.
- `signal_model`: an optional forward-model factor `M` composed into the encoding operator as
  `𝒜 ∘ M`, so the objective becomes `½‖𝒜(M m) − y‖₂² + R(m)` and the optimization variable
  becomes the *model parameter* `m` rather than the image. `nothing` (default) means `𝒜` is
  used unchanged. See below.

# The `signal_model` slot

A forward-model factor is **not** a regularizer: it changes what `𝒜` is, not what is added to
the objective. The `Regularization` interface (`materialize` returning a `Term`,
`scale_regularization`, `get_affected_dims`) cannot express it — `scale_regularization` on a
phase-demodulation operator has no meaning — so it gets its own field.

Concrete signal models and the paradigms they unlock:

| Signal model              | Composition           | Variable            | Enables                        |
| :------------------------ | :-------------------- | :------------------ | :----------------------------- |
| `nothing`                 | `𝒜`                  | image `x`           | standard CS-SENSE              |
| `PhaseDemodulation(ϕ₀)`   | `𝒜 ∘ diag(e^{iϕ₀})`  | real image `m`      | phase-constrained recon — *not built; shipped as the direct `PhaseConstrained` instead* |
| `TemporalBasis(Φ)`        | `𝒜 ∘ Φ`              | coefficients `α`    | subspace recon (§2.7.1)        |
| `OffResonance(ΔB₀, R₂*)`  | time-segmented `𝒜`   | image `x`           | B₀ correction (§2.4.1)         |

All four are linear, so the problem stays convex and every existing solver applies. Non-linear
signal models (quantitative parameter mapping) require the Phase 5 solver work and are out of
scope for this field as specified here.
"""
struct IterativeReconstruction{R, A, D <: ReconstructionDomain, F <: DataFidelity, M} <: AbstractIterativeMethod
    regularization::R
    algorithm::A
    domain::D
    data_fidelity::F
    signal_model::M
end

function IterativeReconstruction(
        regularization::Union{Regularization, Component, Tuple{Vararg{Union{Regularization, Component}}}} = ();
        algorithm = DEFAULT_ALGORITHMS,
        domain::ReconstructionDomain = ImageDomain(),
        data_fidelity::DataFidelity = L2Loss(),
        signal_model = nothing,
    )
    regs = ensure_tuple(regularization)
    check_regularization_shape(regs)   # reuses the existing mixed-tuple rejection
    return IterativeReconstruction(regs, algorithm, domain, data_fidelity, signal_model)
end

# Convenience varargs form. NOTE: the first argument is deliberately *required*. Writing this as
# `IterativeReconstruction(regs::Union{Regularization,Component}...; ...)` would generate a method
# with signature `Tuple{Type{IterativeReconstruction}}` identical to the one the keyword form's
# default argument already generates, silently overwriting it.
IterativeReconstruction(reg1::Union{Regularization, Component}, regs::Union{Regularization, Component}...; kwargs...) =
    IterativeReconstruction((reg1, regs...); kwargs...)
```

Note that `domain` and `data_fidelity` are now **type parameters**, so `_reconstruct_dispatch(acq, ::IterativeReconstruction{<:Any, <:Any, <:KSpaceDomain}, …)` is a real method rather than a runtime branch — matching how `reconstruct.jl` already dispatches on the shape of the regularization tuple.

---

### 5.5 Regularizers Across Domains

Under `KSpaceDomain`, the optimization variable is multi-channel k-space, but most useful regularizers (wavelets, TV, low-rank) are defined on images. The literature formulations resolve this by composing with an inverse Fourier transform: SPIRiT's sparsity term is $\|\Psi \mathcal{F}^{-1} k\|_1$. The API must say where that $\mathcal{F}^{-1}$ comes from.

**Rule: auto-wrap by declared natural domain, with an explicit escape hatch.**

```julia
"""
    natural_domain(reg) -> ReconstructionDomain

The domain in which a regularization term is naturally defined. Image-space priors
(wavelets, TV, LLR, TGV, …) return `ImageDomain()`; k-space structural terms
(`SPIRiTConsistency`, `HankelRankLimit`, `NullSpaceConsistency`) return `KSpaceDomain()`.

When a term's natural domain differs from the reconstruction domain, `reconstruct` composes it
with the transform between them. Under `KSpaceDomain`, an `ImageDomain` regularizer `R` is
applied as `R(ℱ⁻¹ k)` — a **per-coil** inverse Fourier transform, with **no coil combination**,
matching the joint-sparsity-across-coils formulation used by SPIRiT.
"""
natural_domain(::Regularization) = ImageDomain()
```

Two explicit wrappers override the automatic behavior, for the cases where the default is wrong:

```julia
"""
    InImageDomain(reg)

Apply `reg` in the image domain. Explicit form of the default behavior — useful for making
intent visible in code that mixes domains.

    InKSpace(reg)

Apply `reg` directly to the raw k-space samples, bypassing the automatic `ℱ⁻¹` composition.
Needed for terms that genuinely act on k-space values (e.g. an ℓ₁ penalty on k-space itself,
or a per-sample weighting), which would otherwise be silently transformed.
"""
struct InImageDomain{R} <: Regularization; inner::R; end
struct InKSpace{R}      <: Regularization; inner::R; end

natural_domain(::InImageDomain) = ImageDomain()
natural_domain(::InKSpace)      = KSpaceDomain()
```

**Trade-off, stated plainly:** auto-wrapping makes `SPIRiT(; regularization = (L1Wavelet2D(λ),))` read exactly like the paper, at the cost of an implicit transform. The `natural_domain` trait keeps that implicitness *inspectable* (a user can query it) and *overridable* (via the wrappers), which is why it is preferred over either extreme.

**Restriction (finding V4).** "Any image prior auto-wraps" is *false*. `materialize` returns an
opaque `Term`; the only composable seam is `get_operator`, which several terms lack in a usable
form (`PlugAndPlay` has a custom prox, `TotalGeneralizedVariation2D` introduces auxiliary
variables, `MultiScaleLowRank` is a `ProximalAverage`, `RankLimit`/`HardThreshold` are
prox-of-`x` forms). Auto-wrap is therefore **opt-in** via a trait
`is_operator_composable(reg) -> Bool`, `true` only for the wavelet / TV / L1 / LLR family; every
other term under `KSpaceDomain` errors informatively and names `InKSpace(reg)`. `natural_domain`
still declares intent, but `is_operator_composable` gates whether the wrap is actually possible.

**Status:** none of `natural_domain` / `is_operator_composable` / `InImageDomain` / `InKSpace`
exists yet — `domain = KSpaceDomain()` is currently inert. Tracked in `IMPLEMENTATION_PLAN.md`
(post-review follow-up, Part 1).

---

### 5.6 Classification of Iterative Methods

**As-built status (2026-08).** Rows marked *(built)* exist and are tested. Everything else is a
design target: `PhaseDemodulation`, `HankelRankLimit`, `LORAKSRankPenalty`, `NullSpaceConsistency`
and the SAKE / LORAKS / PRUNO presets are **not implemented**. `Phase-Constrained Recon` shipped
instead as the *direct* method `PhaseConstrained` (Margosian, CG on the real-image normal
equations — see §5.7), not as a `signal_model`. `SPIRiT` shipped as a *direct* fixed-point
method; `SPIRiTConsistency` + `KSpaceDomain` (the k-space variational form in this table) are the
post-review follow-up (`IMPLEMENTATION_PLAN.md`, Part 1).

| Method | Domain | Data Fidelity | Mathematical Formulation | Regularizer / Constraint | Solver | Convex? |
| :--- | :---: | :---: | :--- | :--- | :--- | :---: |
| **CS-SENSE / Regularized Recon** *(built)* | image | `L2Loss` | $\min_x \tfrac{1}{2}\|\mathcal{A}x - y\|_2^2 + \lambda \|\Psi x\|_1$ | `L1Wavelet`, `TotalVariation`, `LLR` | `FISTA`, `ADMM` | ✅ |
| **Unregularized iterative LS** *(built)* | image | `L2Loss` | $\min_x \tfrac{1}{2}\|\mathcal{A}x - y\|_2^2$ | — | `CG`, `CGNR` | ✅ |
| **$L+S$ Decomposition** *(built)* | image | `L2Loss` | $\min_{L,S} \tfrac{1}{2}\|\mathcal{A}(L+S) - y\|_2^2 + \|L\|_* + \lambda \|\mathcal{F}_t S\|_1$ | `Component(:L, LowRank)`, `Component(:S, …)` | `ADMM`, `FISTA` | ✅ |
| **Subspace / T2-Shuffling** *(built)* | image | `L2Loss` | $\min_\alpha \tfrac{1}{2}\|\mathcal{A}\Phi\alpha - y\|_2^2 + \lambda R(\alpha)$ | any; `signal_model = TemporalBasis(Φ)` | `FISTA`, `ADMM` | ✅ |
| **Phase-Constrained Recon** *(built as direct `PhaseConstrained`)* | image | `L2Loss` | $\min_{m \in \mathbb{R}} \tfrac{1}{2}\sum_c\|\mathcal{P}\mathcal{F}(s_c e^{i\phi_c} m) - y_c\|_2^2$ | none (direct); CG on normal eqns | — | ✅ |
| **POCS (Partial Fourier)** *(built as direct `POCS`)* | image | consistency ∧ phase | alternating projection, no λ (Haacke 1991) | — | fixed-point | ✅ |
| **SPIRiT** *(direct built; k-space variational form = Part 1)* | k-space | `L2Loss` | $\min_k \tfrac{1}{2}\|\mathcal{P}k - y\|_2^2 + \tfrac{\lambda}{2}\|(I - G)k\|_2^2$ | `SPIRiTConsistency(G)` | `DouglasRachford`, `CGNR` | ✅ |
| **SAKE** *(not implemented)* | k-space | `HardConsistency` | $\min_k\; i_{\{\operatorname{rank}\mathcal{H}(k)\,\le\, r\}}(k) + i_{\{\mathcal{P}k = y\}}(k)$ | `HankelRankLimit(kernel_size, rank)` | `DouglasRachford` | ❌ |
| **LORAKS / AC-LORAKS** *(not implemented)* | k-space | `L2Loss` | $\min_k \tfrac{1}{2}\|\mathcal{P}k - y\|_2^2 + \lambda\, J_r(\mathcal{C}(k))$ | `LORAKSRankPenalty(r)` | `ADMM`, `DouglasRachford` | ❌ |
| **PRUNO** *(not implemented)* | k-space | `HardConsistency` | $\min_k \tfrac{1}{2}\|N k\|_2^2 + i_{\{\mathcal{P}k = y\}}(k)$ | `NullSpaceConsistency(N)` | `DouglasRachford` | ✅ |

**Corrections relative to the previous draft, and why they matter:**

* **POCS** previously wrote $i_{\{\mathcal{F}x = y\}}$. Enforcing consistency against the *full* Fourier transform over-constrains the problem — partial Fourier data does not determine unacquired samples. Consistency holds only on acquired samples, hence $\mathcal{A}$ (or $\mathcal{D}$ in k-space form). The phase set is written explicitly as the real ray $\{x = e^{i\phi_0}m,\, m \in \mathbb{R}\}$, which is a convex subspace; "$\arg(x) = \phi_0$" is ambiguous at $x = 0$.
* **SAKE** previously used the nuclear norm $\|\mathcal{H}(k)\|_*$ in the table while the accompanying preset used a hard rank limit. The published algorithm is hard-rank (Cadzow); the table now matches the code, and the non-convexity is flagged.
* **LORAKS** previously wrote $\min \operatorname{rank}(\mathcal{C}(k))$, which is not a proximal-solvable objective and is not what LORAKS does. Replaced with the penalized form from Haldar (2014).
* **PRUNO** previously combined `data_fidelity = :none` with a `HardDataConstraint()` term in the regularizer list — the same constraint expressed twice, in two different places. It now uses `HardConsistency`, which is what that combination meant. This redundancy is the clearest evidence that fidelity belongs on its own axis rather than as an ordinary term.
* **Solver column** no longer lists `AFBA`, which MRT does not expose. `DouglasRachford` **is now aliased, exported and patched** (`patch_algorithm_with_default_values` supplies `gamma`); it is part of `DEFAULT_ALGORITHMS`. `HardConsistency` is general via an inner CG when `𝒜𝒜'` is not diagonal, not restricted to the diagonal case.

---

### 5.7 Non-Iterative Methods and Named Presets

#### A. Direct Adjoint Reconstruction

```julia
"""
    DirectReconstruction(; dcf = nothing)

Direct adjoint reconstruction (zero-filled IFFT or NFFT gridding, `x̂ = 𝒜' * y`).

- `dcf`: density compensation weights applied before the adjoint, giving `x̂ = 𝒜' * (w .* y)`.
  Either an array of weights, or a `DensityCompensation` strategy (`PipeMenonDCF()`,
  `VoronoiDCF()`) computed from the trajectory. `nothing` (default) applies no weighting, which
  is the current behavior and is badly conditioned for non-Cartesian trajectories (§2.3).
"""
Base.@kwdef struct DirectReconstruction{D} <: AbstractDirectMethod
    dcf::D = nothing
end
```

#### B. Non-Iterative Homodyne

```julia
"""
    Homodyne(; symmetric_band = nothing, ramp = LinearRamp(), readout_dim = nothing)

Non-iterative homodyne detection: asymmetric ramp weighting in k-space, low-resolution phase
estimation, inverse transform, phase demodulation, and real part.

- `symmetric_band`: width (in samples) of the fully sampled symmetric band used for phase
  estimation. `nothing` (default) derives it from the acquisition's sampling pattern, which is
  the only way to get it right — partial Fourier is asymmetric along exactly one encoding
  direction, and the band width is determined by the sampled fraction.
- `readout_dim`: which encoding dimension is partially sampled; `nothing` infers it.
- `ramp`: `LinearRamp()` or `StepRamp()`.

Discards residual phase; see `PhaseConstrained` and `POCS` for phase-preserving variants.
"""
Base.@kwdef struct Homodyne{S, R} <: AbstractDirectMethod
    symmetric_band::S = nothing
    readout_dim::Union{Nothing, Int, Symbol} = nothing
    ramp::R = LinearRamp()
end
```

#### C. GRAPPA

```julia
"""
    GRAPPA(; kernel_size = (4, 5), calib_size = (24, 24), λ = 1e-4)

Non-iterative k-space convolution kernel synthesis (Griswold 2002). `λ` is the Tikhonov
regularization used when solving the kernel-fitting least-squares problem against the ACS data.

Applicable only to uniformly undersampled Cartesian acquisitions with an ACS region; this is
checked by `check_applicable` before any work is done (§5.9).
"""
Base.@kwdef struct GRAPPA{K, C, T} <: AbstractDirectMethod
    kernel_size::K = (4, 5)
    calib_size::C = (24, 24)
    λ::T = 1.0e-4
end
```

#### D. Named Iterative Presets

Presets are **types**, not functions returning `IterativeReconstruction`. A function-based preset
would mean `SPIRiT` is not a type, cannot be dispatched on, `SPIRiT(...) isa AbstractReconstructionMethod`
is only incidentally true, and both `show` output and error messages lose the user's intent —
a mis-specified SPIRiT reconstruction would report itself as a generic `IterativeReconstruction`.
Each preset carries its own fields and implements `lower`, so exactly one execution path remains.

```julia
"""
    POCS(; symmetric_band = nothing, algorithm = (DouglasRachford(),))

Phase-constrained partial Fourier reconstruction by projection onto convex sets.
"""
Base.@kwdef struct POCS{S, A} <: AbstractIterativeMethod
    symmetric_band::S = nothing
    algorithm::A = (DouglasRachford(),)
end

lower(m::POCS) = IterativeReconstruction(
    PhaseConstraint(; symmetric_band = m.symmetric_band);
    algorithm = m.algorithm, domain = ImageDomain(), data_fidelity = HardConsistency(),
)

"""
    PhaseConstrained(; coil_combination = AdjointSensitivity())

Partial Fourier reconstruction optimizing a real-valued image under an estimated phase map
(Margosian et al. 1986).

As built: `PhaseConstrained <: AbstractDirectMethod` (not an iterative preset). The low-resolution
phase is estimated from the symmetric centre and the real-valued least-squares problem
`min_{m ∈ ℝ} Σ_c ‖𝒫 ℱ (s_c e^{iφ_c} m) − y_c‖²` is solved by conjugate gradient on the normal
equations. `_direct_reconstruct(acq, ::PhaseConstrained)` in `methods/partial_fourier.jl`.
"""
struct PhaseConstrained{C <: CoilCombination} <: AbstractDirectMethod
    coil_combination::C
end

"""
    SPIRiT(; kernel_size = (5, 5), calib_size = (24, 24), regularization = (), algorithm = (CGNR(),))

Iterative self-consistent parallel imaging (Lustig & Pauly 2010). Image-domain regularizers are
composed with a per-coil `ℱ⁻¹` automatically (§5.5).
"""
Base.@kwdef struct SPIRiT{K, C, R, A} <: AbstractIterativeMethod
    kernel_size::K = (5, 5)
    calib_size::C = (24, 24)
    regularization::R = ()
    algorithm::A = (CGNR(),)
end

lower(m::SPIRiT) = IterativeReconstruction(
    (SPIRiTConsistency(; kernel_size = m.kernel_size, calib_size = m.calib_size), ensure_tuple(m.regularization)...);
    algorithm = m.algorithm,
    domain = KSpaceDomain(; coil_combine = AdjointSensitivity()),
    data_fidelity = L2Loss(),
)

"""
    SAKE(; kernel_size = (5, 5), rank = 12, algorithm = (DouglasRachford(),))

Calibrationless structured low-rank matrix completion (Shin 2014).

!!! warning "Non-convex"
    The fixed-rank set is non-convex, so the splitting algorithm is a heuristic here: it may
    converge to a non-global fixed point or fail to converge, and the result depends on the
    initial estimate. This mirrors the original Cadzow-style alternating-projection algorithm.
"""
Base.@kwdef struct SAKE{K, T, A} <: AbstractIterativeMethod
    kernel_size::K = (5, 5)
    rank::T = 12
    algorithm::A = (DouglasRachford(),)
end

lower(m::SAKE) = IterativeReconstruction(
    HankelRankLimit(; kernel_size = m.kernel_size, rank = m.rank);
    algorithm = m.algorithm,
    # Calibrationless by construction: no sensitivity maps are assumed to exist.
    domain = KSpaceDomain(; coil_combine = RootSumSquares()),
    data_fidelity = HardConsistency(),
)
```

---

### 5.8 `reconstruct` Signatures

```julia
"""
    reconstruct(acq_data::AcquisitionInfo, method::AbstractReconstructionMethod = DirectReconstruction(); kwargs...)

Perform image reconstruction on `acq_data` using the specified `method`.

# Keyword arguments
- `x₀`: initial guess. Its expected shape follows the method's domain: an image-sized array for
  `ImageDomain`, a multi-channel k-space array for `KSpaceDomain`, and a `Tuple`/`NamedTuple` of
  image-sized arrays when the method's regularization is a set of `Component`s.
- all existing `Config` keywords (`tol`, `maxit`, `verbose`, `threaded`, `normalization`,
  `disable_task_splitting`, …) are unchanged.
"""
function reconstruct(
        acq_data::AcquisitionInfo,
        method::AbstractReconstructionMethod = DirectReconstruction();
        x₀ = nothing,
        kwargs...,
    )
    config = construct_config(kwargs)
    method = lower(method)
    check_applicable(method, acq_data)
    check_x₀_shape(x₀, method, acq_data)
    t_start = time()
    x = _reconstruct_dispatch(acq_data, method, x₀, config)
    t_end = time()
    config.verbose && config.printfunc("Total time: ", format_time(t_end - t_start))
    return x
end
```

**Deprecation path.** The three-positional-argument signature is the current public API and is used
throughout the test suite and documentation. Replacing it outright produces a bare `MethodError`
with no indication of what to do. A shim in the style of the existing mixed-tuple error method in
`reconstruct.jl` keeps old code working for one release cycle:

```julia
reconstruct(acq_data, IterativeReconstruction(regularization; algorithm); kwargs...)
```

As built: **no deprecation shim was added** — the old positional
`reconstruct(acq_data, regularization, algorithm)` form is gone entirely (the package had no
users). The only surviving signature is the `method`-based one above.

`reconstruct(acq_data)` with no method still returns the **direct adjoint**: the default method is
`DirectReconstruction()`, and the `regularization == ()` short-circuit became
`method isa AbstractDirectMethod`.

---

### 5.9 Applicability Checking

Several methods apply only to particular acquisitions: GRAPPA needs uniform Cartesian
undersampling with ACS; `Homodyne`/`POCS` need a partial Fourier pattern; `KSpaceDomain` needs
multi-channel data; `AdjointSensitivity` coil combination needs sensitivity maps. Without an
explicit check, a mismatch surfaces as a shape error deep inside an operator, long after the
scaling and planning work has been done.

```julia
"""
    check_applicable(method, acq_data)

Throw an informative `ArgumentError` if `method` cannot be applied to `acq_data`. Called by
`reconstruct` before any operator construction. Defaults to a no-op.
"""
check_applicable(::AbstractReconstructionMethod, ::AcquisitionInfo) = nothing

function check_applicable(m::GRAPPA, acq::AcquisitionInfo)
    @argcheck acq isa CartesianAcquisitionInfo "GRAPPA requires Cartesian data; got $(typeof(acq)). Use SPIRiT for non-Cartesian k-space interpolation."
    @argcheck has_uniform_undersampling(acq) "GRAPPA requires uniform undersampling along the phase-encoding direction(s)."
    @argcheck has_acs_region(acq, m.calib_size) "GRAPPA requires a fully sampled ACS region of at least $(m.calib_size)."
    return nothing
end
```

This mirrors how `check_components` already validates component sets up front rather than letting a
mistyped name surface later.

---

### 5.10 Task Splitting

Automatic task splitting integrates with the method and its domain:

* **`DirectReconstruction`**: splits over all non-Fourier batch dimensions.
* **`IterativeReconstruction{…, ImageDomain}`**: splits over image batch dimensions unaffected
  by any regularizer — unchanged from today.
* **`IterativeReconstruction{…, KSpaceDomain}`**: splits over batch dimensions (slices,
  contrasts) while keeping $(k_x, k_y, N_c)$ intact on each subproblem. The **coil dimension must
  never be split**, since every k-space method couples channels by construction.
* **`GRAPPA` / `Homodyne`**: split across batch dimensions, performing calibration and synthesis
  per slice.
* **`signal_model` interaction**: a signal model couples the dimensions it acts on, exactly as a
  regularizer does. `TemporalBasis(Φ)` couples the temporal dimension and forbids task splitting
  over it; `OffResonance` couples nothing extra. The task-splitting planner therefore needs
  `get_affected_dims(signal_model, …)` alongside the existing regularizer query — this is a small
  but easy-to-miss extension of `get_task_splitting_plan`.
* **SMS caveat (future)**: simultaneous multi-slice breaks the assumption that slices are separable
  subproblems. If SMS is ever added, the task-splitting planner must be told that the slice
  dimension is coupled.

---

### 5.11 Open Questions

1. **Scope of learned reconstruction.** `PlugAndPlay` fits the current architecture; unrolled
   networks (MoDL, VarNet) need a training loop and autodiff through the forward operator, which is
   a different execution model. Does this belong in MRT, in a downstream package, or nowhere?
2. **Non-linear solver strategy.** JSENSE/NLINV, quantitative mapping and graph-cut water/fat all
   need machinery outside `StructuredOptimization.jl`'s linear-operator term algebra. Options are an
   IRGNM implementation inside MRT, an alternating-minimization layer over the existing solvers, or
   a dependency on a general non-linear optimizer. This choice shapes all of Phase 5 and should be
   made before Phase 4 concludes.
3. **`AcquisitionInfo` versus method for model factors.** `signal_model` places $B_0$ maps and
   temporal bases on the *method*. An arguable alternative is to place them on `AcquisitionInfo`,
   since they describe the physics of the acquisition rather than a reconstruction choice. The
   current proposal keeps them on the method so that the same acquisition can be reconstructed with
   and without correction, but the boundary is worth revisiting once $B_0$ correction lands.
4. **`algorithm` placement.** The solver currently lives inside the method, which keeps presets
   self-contained but makes `algorithm` a per-method field while `x₀`, `tol` and `maxit` are
   `reconstruct` keywords. An alternative is to move `algorithm` to a `reconstruct` keyword that
   overrides the method's default. Low stakes, but it should be decided once rather than drifting.
