# Large datasets

MRT alone on two full-size scanner datasets, at 1, 4 and 8 host threads and on one GPU. The catalog
cases of `benchmark/comparison/` are small enough that every toolkit finishes them; these are not,
and they show how MRT scales to a full 3D volume and to a full multislice non-Cartesian scan.

| case | data | size |
|---|---|---|
| `large_3d_knee_cartesian` | MRIDATA knee `52c2fd53-d233-4444-8bfd-7c454240d314`, the whole volume | 320×320×256, 8 coils, ky-kz variable density, R = 6 |
| `large_multislice_breast_radial` | fastMRI breast `fastMRI_breast_006_2`, every partition of the stack of stars | 83 slices of 320², 16 coils, 144 of 288 golden-angle spokes × 320 samples |

The methods and their effort are the comparison suite's (`benchmark/utils/mrt_methods.jl`):
CG-SENSE (10 CGNR iterations), L1-wavelet (20 FISTA iterations), TV (20 ADMM iterations of 10 CG
each) and anisotropic TV by PDHG (200 iterations), λ from the synthetic analogue
(`shepp_logan_3d_8ch_cartesian`, `shepp_logan_2d_8ch_radial`). Each row is compiled on the
analogue case and then timed once, from host data to host image. See `run.jl`'s header for how
each case is prepared.

```sh
benchmark/slurm/submit.sh large_datasets.sh --threads=16 --prepare      # once: build the case cache
benchmark/slurm/submit.sh --production large_datasets.sh --threads=1,4,8
benchmark/slurm/submit.sh --production large_datasets_gpu.sh
```

`large_datasets.sh` books one node and runs the thread counts at the same time, each on the
physical cores of its own NUMA domain. Results land in `results/<backend>.json`, one row per case
and method.
