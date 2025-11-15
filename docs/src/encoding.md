# API Reference

This page documents all public functions and types in MriReconstructionToolbox.jl for creating encoding operators that model the complete MRI acquisition process.

## Encoding Operators

The main interface for creating composite operators that model the complete MRI acquisition chain.

```@docs
MriReconstructionToolbox.get_encoding_operator
```

## Fourier Operators

Functions for creating operators that transform between image space and k-space using Fast Fourier Transforms.

```@docs
MriReconstructionToolbox.get_fourier_operator
```

## Sensitivity Map Operators

Functions for creating operators that model multi-coil parallel imaging with coil sensitivity maps.

```@docs
MriReconstructionToolbox.get_sensitivity_map_operator
```

## Subsampling Operators

Functions for creating operators that handle undersampled k-space data for accelerated imaging.

```@docs
MriReconstructionToolbox.get_subsampled_fourier_op
```

## Types

Type definitions for subsampling patterns used in accelerated MRI acquisition.

```@docs
MriReconstructionToolbox._1D_subsampling_type
MriReconstructionToolbox._2D_subsampling_type
MriReconstructionToolbox._3D_subsampling_type
```
