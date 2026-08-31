module Phantoms

using FFTW
using GeometricMedicalPhantoms

"""
    generate_multicoil_brain(; N=64, num_coils=8)

Generates a simple 2D phantom using GeometricMedicalPhantoms and simulates multi-coil sensitivities.
Returns `(image, kspace, sensitivities)`.
"""
function generate_multicoil_brain(; N=64, num_coils=8)
    # Generate brain phantom from GeometricMedicalPhantoms
    img = create_shepp_logan_phantom(N, N, :axial)
    
    fov = 20.0 # Default FOV in create_shepp_logan_phantom
    x = range(-fov/2, fov/2, length=N)
    y = range(-fov/2, fov/2, length=N)
    
    # Generate mock coil sensitivities
    sens = zeros(ComplexF64, N, N, num_coils)
    for c = 1:num_coils
        phase = 2π * (c - 1) / num_coils
        for j=1:N, i=1:N
            cx = (fov/2 * 0.8) * cos(phase)
            cy = (fov/2 * 0.8) * sin(phase)
            dist2 = (x[i] - cx)^2 + (y[j] - cy)^2
            sens[i, j, c] = exp(-dist2 / (fov^2 * 0.25)) * exp(1im * phase)
        end
    end
    
    # Coil images
    coil_imgs = img .* sens
    
    # K-space
    kspace = fftshift(fft(ifftshift(coil_imgs, (1, 2)), (1, 2)), (1, 2))
    
    return img, kspace, sens
end

export generate_multicoil_brain

end
