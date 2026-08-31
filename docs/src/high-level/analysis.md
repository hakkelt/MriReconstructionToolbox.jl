# Noise and Reconstruction Analysis

`MriReconstructionToolbox` provides analysis tools for evaluating noise propagation, SNR maps, and g-factor geometry in MRI reconstructions.

## Pseudo-Replica Noise Propagation

The Monte Carlo pseudo-replica method (Robson et al. 2008) measures pixel-by-pixel noise variance and parallel imaging geometry factors ($g$-factor) for arbitrary non-linear and regularized reconstruction pipelines.

```@docs
pseudo_replica
```

### Usage

```julia
# Run 64 pseudo-replica iterations with fixed scaling
res = pseudo_replica(acq_data, method; replicas = 64, normalization = NoScaling())

mean_img = res.mean
std_img = res.std
g_factor_map = res.g_factor
```

> [!IMPORTANT]
> `pseudo_replica` requires `normalization = FixedScaling(...)` or `normalization = NoScaling()`. Data-dependent percentile scaling (`BartScaling()`) rescales each noisy replica independently by its own noise quantile, distorting inter-replica variance.
