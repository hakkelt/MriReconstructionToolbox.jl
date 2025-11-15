"""
    coil_sensitivities(nx::Int, ny::Int, nc::Int) -> Array{ComplexF32, 3}
    coil_sensitivities(nx::Int, ny::Int, nz::Int, nc::Int) -> Array{ComplexF32, 4}

Generate simulated sensitivity maps for multi-coil MRI receivers.

# Physics Background
In parallel MRI, multiple receiver coils are arranged around the imaging subject, each with
spatially varying sensitivity profiles. This function simulates realistic coil sensitivity
maps based on the physical principles of electromagnetic reception.

# Physical Model
The sensitivity maps are constructed using:

1. **Spatial Arrangement**: Coils are positioned in a circular array around the field of view,
   with centers at angles `2π(i-1)/nc` for coil `i`, mimicking typical clinical coil arrays.

2. **Magnitude Profile**: Each coil's sensitivity decreases with distance from its center
   following a Gaussian profile `exp(-r²/(2σ²))`, where `r` is the distance from the coil
   center. This reflects the physical reality that receiver coils are most sensitive to signal
   sources near them, with sensitivity falling off smoothly with distance.

3. **Phase Variation**: A linear phase ramp `exp(im(0.5x + 0.3y))` is applied to simulate
   phase variations due to:
   - B₀ field inhomogeneities
   - Receiver electronics phase offsets
   - Geometric positioning effects

4. **Normalization**: The maps are normalized using the root-sum-of-squares across all coils,
   ensuring that `√(Σᵢ|sᵢ(x,y)|²) ≈ 1` at each spatial location. This preserves signal
   intensity while maintaining spatial encoding information.

# Arguments
- `nx::Int`: Number of pixels in x-direction
- `ny::Int`: Number of pixels in y-direction
- `nz::Int`: Number of pixels in z-direction (for 3D sensitivities)
- `nc::Int`: Number of coils

# Returns
- `Array{ComplexF32, 3}`: Sensitivity maps of size (nx, ny, nc), where each slice along
  dimension 3 represents one coil's complex-valued sensitivity profile.

"""
function coil_sensitivities(nx::Int, ny::Int, nc::Int)
    x = range(-1, 1; length=nx)
    y = range(-1, 1; length=ny)
    X = repeat(collect(x), 1, ny)
    Y = repeat(collect(y)', nx, 1)
    smaps = Array{ComplexF32}(undef, nx, ny, nc)
    centers = [(cos(2π*(i-1)/nc), sin(2π*(i-1)/nc)) for i in 1:nc]
    for i in 1:nc
        cx, cy = centers[i]
        σ = 0.6f0
        mag = @. exp(-((X - cx)^2 + (Y - cy)^2) / (2σ^2))
        phase = @. exp(im * (0.5f0 * X + 0.3f0 * Y))
        smaps[:, :, i] = ComplexF32.(mag .* phase)
    end
    denom = sqrt.(sum(abs2, smaps; dims=3) .+ eps(Float32))
    smaps ./= denom
    return smaps
end

function coil_sensitivities(nx::Int, ny::Int, nz::Int, nc::Int)
    sm2d = reshape(coil_sensitivities(nx, ny, nc), nx, ny, 1, nc)
    return repeat(sm2d, 1, 1, nz, 1)
end