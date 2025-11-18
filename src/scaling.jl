abstract type Normalization end

"""
    NoScaling() <: Normalization

A normalization strategy that applies no scaling to the data.
"""
struct NoScaling <: Normalization end

"""
    BartScaling() <: Normalization

A normalization strategy that mimics the scaling used in BART.
This approach inspects the distribution of the absolute values of the initial guess `x₀`
(obtained as the adjoint of the encoding operator applied to the measured k-space data)
and selects either the 90th percentile or the maximum value, depending on the spread of the values.
If the difference between the maximum and the 90th percentile is less than twice the difference
between the 90th percentile and the median, the 90th percentile is used; otherwise, the maximum value is used.
This helps to avoid scaling based on outliers in the data.
"""
struct BartScaling <: Normalization end

"""
    MeasurementBasedScaling() <: Normalization

A normalization strategy that scales the data based on the average absolute value of the k-space measurements.
This approach mimic the normalization of MeasurementBasedNormalization from RegularizedLeastSquares.jl
(which is used by MRIReco.jl).
"""
struct MeasurementBasedScaling <: Normalization end

"""
    get_scale(scaling::Normalization, acq_data::AcquisitionInfo, x₀)

Computes the scaling factor based on the chosen normalization strategy.

# Arguments
- `scaling::Normalization`: The normalization strategy to use (NoScaling, BartScaling, MeasurementBasedScaling, or FixedScaling).
- `acq_data::AcquisitionInfo`: The acquisition information containing k-space data and other parameters.
- `x₀::AbstractArray`: The initial guess for the image, typically obtained as the adjoint of the encoding operator applied to the k-space data.

# Returns
- `scale::Float64`: The computed scaling factor.
"""
function get_scale(::NoScaling, acq_data::AcquisitionInfo, x₀)
    return 1.0
end

function get_scale(::BartScaling, acq_data::AcquisitionInfo, x₀)
    median, p90, max = quantile(abs.(vec(x₀)), [0.5, 0.9, 1.0])
    return ((max - p90) < 2 * (p90 - median)) ? p90 : max
end

function get_scale(::MeasurementBasedScaling, acquisition_data::AcquisitionInfo, x₀)
    return norm(acquisition_data.kspace_data, 1) / length(acquisition_data.kspace_data)
end
