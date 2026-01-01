"""
    create_torso_phantom(nx::Int=128, ny::Int=128, nz::Int=128; fovs=(30, 24, 30)) -> Array{ComplexF32, 3}

Generate a 3D torso phantom with anatomical structures including torso outline, lungs, heart, and vessels.

# Arguments
- `nx::Int=128`: Number of voxels in x-direction
- `ny::Int=128`: Number of voxels in y-direction  
- `nz::Int=128`: Number of voxels in z-direction

# Keywords
- `fovs::Tuple=(30, 24, 30)`: Field of view in cm for (x, y, z) directions

# Returns
- Array{ComplexF32, 3}: 3D phantom array with size (nx, ny, nz)

# Description
Creates a simplified anatomical torso phantom with the following structures:
- Torso: Multi-segment outer boundary (upper/middle/lower) for realistic shape (intensity 0.2)
- Lungs: Multiple ellipsoids representing left/right lung lobes (intensities 0.08-0.1)
- Heart: Multiple ellipsoids for ventricles and atria (intensities 0.6-0.7)
- Vessels: Aorta, pulmonary artery, and superior vena cava (intensities 0.75-0.8)
- Spine: Vertebral column with 12 vertebrae (intensity 0.9)
- Ribs: Paired rib structures arranged in 9 levels (intensity 0.85)

# Example
```julia
phantom = create_torso_phantom(128, 128, 128)
```
"""
function create_torso_phantom(nx::Int=128, ny::Int=128, nz::Int=128; fovs=(30, 30, 30))
    # Coordinate axes
    Δx, Δy, Δz = fovs[1]/nx, fovs[2]/ny, fovs[3]/nz
    ax_x = range(-(nx-1)/2, (nx-1)/2, length=nx) .* Δx
    ax_y = range((ny-1)/2, -(ny-1)/2, length=ny) .* Δy
    ax_z = range(-(nz-1)/2, (nz-1)/2, length=nz) .* Δz

    # Normalize to [-1, 1] range for easier ellipsoid definitions
    x_range = maximum(ax_x) - minimum(ax_x)
    y_range = maximum(ax_y) - minimum(ax_y)
    z_range = maximum(ax_z) - minimum(ax_z)
    ax_xn = @. 2 * ax_x / x_range
    ax_yn = @. 2 * ax_y / y_range
    ax_zn = @. 2 * ax_z / z_range

    # Initialize phantom
    phantom = zeros(Float32, nx, ny, nz)

    # Add anatomical structures in order (background to foreground)
    add_torso_boundary!(phantom, ax_xn, ax_yn, ax_zn)
    add_arm_bones_axes!(phantom, ax_xn, ax_yn, ax_zn)  # Call before ribs
    add_lungs_axes!(phantom, ax_xn, ax_yn, ax_zn)
    add_heart_axes!(phantom, ax_xn, ax_yn, ax_zn)
    add_vessels_axes!(phantom, ax_xn, ax_yn, ax_zn)
    add_spine_axes!(phantom, ax_xn, ax_yn, ax_zn)
    add_ribs_axes!(phantom, ax_xn, ax_yn, ax_zn)
    add_liver_axes!(phantom, ax_xn, ax_yn, ax_zn)
    add_stomach_axes!(phantom, ax_xn, ax_yn, ax_zn)

    return ComplexF32.(phantom)
end

# === Core drawing primitive ===
# Draw a superellipsoid defined by center (cx,cy,cz), radii (rx,ry,rz), exponents (exx, exy, exz)
# onto phantom using normalized axes ax_x, ax_y, ax_z. Avoids creating 3D grids.
function draw_superellipsoid!(phantom::Array{Float32,3}, ax_x::AbstractVector, ax_y::AbstractVector, ax_z::AbstractVector,
                              cx::Real, cy::Real, cz::Real,
                              rx::Real, ry::Real, rz::Real;
                              ex::NTuple{3,Real}=(2.5,2.5,2.5), intensity::Real)
    # Restrict computation to the enclosing axis-aligned box to avoid full-volume work
    @inline function idx_bounds(ax::AbstractVector, c::Real, r::Real)
        n = length(ax)
        # Handle both ascending and descending axes
        step = n > 1 ? (ax[2] - ax[1]) : 1.0
        i1 = 1 + ((c - r) - ax[1]) / step
        i2 = 1 + ((c + r) - ax[1]) / step
        i_min = clamp(Int(floor(min(i1, i2))), 1, n)
        i_max = clamp(Int(ceil(max(i1, i2))), 1, n)
        return i_min, i_max
    end

    ix_min, ix_max = idx_bounds(ax_x, cx, rx)
    iy_min, iy_max = idx_bounds(ax_y, cy, ry)
    iz_min, iz_max = idx_bounds(ax_z, cz, rz)

    inv_rx = 1.0 / rx
    inv_ry = 1.0 / ry
    inv_rz = 1.0 / rz
    exx, exy, exz = ex
    val_int = Float32(intensity)

    @inbounds for i in ix_min:ix_max
        dx = abs(ax_x[i] - cx) * inv_rx
        dxp = dx^exx
        for j in iy_min:iy_max
            dy = abs(ax_y[j] - cy) * inv_ry
            dyp = dy^exy
            for k in iz_min:iz_max
                dz = abs(ax_z[k] - cz) * inv_rz
                dzp = dz^exz
                if dxp + dyp + dzp <= 1.0
                    phantom[i, j, k] = val_int
                end
            end
        end
    end
    return phantom
end

"""
Helper function to add torso outer boundary with shoulders, neck, and arms.
Uses smaller superellipsoids (n=2.5) with merged segments for efficiency.
"""
function add_torso_boundary!(phantom, ax_x, ax_y, ax_z)
    # Neck
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, -0.165, 0.75, 0.312, 0.336, 0.22; ex=(2.5,2.5,2.5), intensity=0.2)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, -0.165, 0.92, 0.312, 0.312, 0.15; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Shoulders
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, -0.165, 0.70, 0.8, 0.28, 0.25; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Arms
    # Left arm - upper
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -0.68, -0.165, 0.62, 0.18, 0.18, 0.28; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Left arm - mid
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -0.88, -0.165, 0.50, 0.17, 0.17, 0.26; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Left arm - lower
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -1.05, -0.165, 0.38, 0.16, 0.16, 0.24; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Right arm - upper
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.68, -0.165, 0.62, 0.18, 0.18, 0.28; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Right arm - mid
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.88, -0.165, 0.50, 0.17, 0.17, 0.26; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Right arm - lower
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 1.05, -0.165, 0.38, 0.16, 0.16, 0.24; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Chest
    # Ribs 1-4 level (upper chest)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.0, 0.45, 0.86, 0.69, 0.35; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Ribs 5-8 level (mid chest - widest)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.0, 0.17, 0.93, 0.72, 0.32; ex=(2.5,2.5,3.5), intensity=0.2)
    
    # Ribs 9-12 level (lower chest)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.0, -0.11, 0.91, 0.71, 0.32; ex=(2.5,2.5,3.5), intensity=0.2)
    
    # Abdomen
    # Upper abdomen
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.0, -0.39, 0.87, 0.67, 0.32; ex=(2.5,2.5,3.5), intensity=0.2)
    
    # Lower abdomen
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.0, -0.85, 0.83, 0.62, 0.45; ex=(2.5,2.5,3.5), intensity=0.2)
    
    # Posterior extensions for spine/back coverage
    # Upper back (cervical/upper thoracic region)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.28, 0.35, 0.70, 0.43, 0.50; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Mid back (mid thoracic region)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.28, -0.10, 0.75, 0.47, 0.55; ex=(2.5,2.5,2.5), intensity=0.2)
    
    # Lower back (lumbar region)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, 0.15, -0.60, 0.78, 0.48, 0.55; ex=(2.5,2.5,2.5), intensity=0.2)
end

"""
Helper function to add lungs with upper lobe, lower lobe, heart cavity, and diaphragm.
"""
function add_lungs_axes!(phantom, ax_x, ax_y, ax_z)
    lung_x_offset = 0.32
    
    # Left Lung
    # Upper lobe
    lung_l_top_x = lung_x_offset - 0.15
    lung_l_top_radius = 0.74 * 0.48
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -lung_l_top_x, 0.0, 0.289, lung_l_top_radius, lung_l_top_radius, 0.4375; ex=(2.5,2.5,1.5), intensity=0.08)
    
    # Lower lobe
    lung_l_lower_radius = 0.80 * 0.48 * 1.05
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -lung_x_offset, 0.0, -0.20, lung_l_lower_radius, lung_l_lower_radius, 0.45; ex=(2.5,2.5,2.5), intensity=0.09)
    
    # Diaphragm
    diaphragm_radius = lung_l_lower_radius * 0.80
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -lung_x_offset, 0.0, -0.70, diaphragm_radius, diaphragm_radius, 0.20; ex=(2.5,2.5,1.5), intensity=0.2)
    
    # Right Lung
    lung_r_top_x = lung_x_offset - 0.15
    # Upper lobe
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, lung_r_top_x, 0.0, 0.289, lung_l_top_radius, lung_l_top_radius, 0.4375; ex=(2.5,2.5,1.5), intensity=0.08)
    # Lower lobe
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, lung_x_offset, 0.0, -0.20, lung_l_lower_radius, lung_l_lower_radius, 0.45; ex=(2.5,2.5,2.5), intensity=0.09)
    # Diaphragm
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, lung_x_offset, 0.0, -0.70, diaphragm_radius, diaphragm_radius, 0.20; ex=(2.5,2.5,1.5), intensity=0.2)
    
    # Heart cavity
    #draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.05, 0.0, 0.0, 0.30, 0.30, 0.50; ex=(2.2,2.2,1.5), intensity=0.2)
end

"""
Helper function to add heart chambers.
"""
function add_heart_axes!(phantom, ax_x, ax_y, ax_z)
    # Heart main body
    # Upper heart (base)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.05, 0.0, -0.1, 0.25, 0.25, 0.20; ex=(2.0,2.0,2.0), intensity=0.6)
    
    # Mid heart
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.05, 0.0, 0.0, 0.23, 0.23, 0.18; ex=(2.5,2.5,2.5), intensity=0.6)
    
    # Lower heart (apex)s parts part    # Lower heart (apex)s parts part

    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.05, 0.0, 0.15, 0.18, 0.18, 0.22; ex=(3.5,3.5,2.0), intensity=0.6)
    
    # Left ventricle
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, -0.02, 0.0, 0.18, 0.18, 0.15; ex=(2.0,2.0,2.0), intensity=0.7)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, -0.02, 0.12, 0.14, 0.14, 0.15; ex=(3.0,3.0,2.0), intensity=0.7)
    
    # Right ventricle
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.10, 0.0, 0.0, 0.15, 0.15, 0.14; ex=(2.0,2.0,2.0), intensity=0.65)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.10, 0.0, 0.12, 0.12, 0.12, 0.13; ex=(3.0,3.0,2.0), intensity=0.65)
    
    # Left atrium
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.03, 0.05, -0.2, 0.12, 0.12, 0.15; ex=(2.2,2.2,2.2), intensity=0.68)
    
    # Right atrium
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.03, 0.07, -0.2, 0.12, 0.12, 0.15; ex=(2.2,2.2,2.2), intensity=0.63)
end

"""
Helper function to add major vessels.
"""
function add_vessels_axes!(phantom, ax_x, ax_y, ax_z)
    function draw_vessel!(x_center, y_center, z_center, radius_xy, height_z, intensity, n)
        draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, x_center, y_center, z_center, radius_xy, radius_xy, height_z; ex=(n,n,n), intensity=intensity)
    end
    
    # Aorta (ascending)
    for (z_center, half_height) in [(0.70, 0.08), (0.55, 0.12), (0.40, 0.12), (0.25, 0.12), (0.10, 0.12)]
        draw_vessel!(-0.02, -0.05, z_center, 0.06, half_height, 0.8, 2.5)
    end
    
    # Pulmonary artery
    for (z_center, half_height) in [(0.60, 0.08), (0.47, 0.10), (0.32, 0.12), (0.17, 0.12), (0.05, 0.10)]
        draw_vessel!(-0.05, -0.05, z_center, 0.05, half_height, 0.75, 2.5)
    end
    
    # Superior vena cava
    for (z_center, half_height) in [(0.85, 0.08), (0.72, 0.10), (0.58, 0.12), (0.45, 0.12), (0.35, 0.08)]
        draw_vessel!(0.1, -0.05, z_center, 0.04, half_height, 0.78, 2.5)
    end
end

"""
Helper function to add spine (vertebral column) with spinal curvature.
"""
function add_spine_axes!(phantom, ax_x, ax_y, ax_z)
    spine_intensity = 0.55
    
    # Spinal curvature function: Y offset based on Z position
    function spine_curve(z)
        if z > 0.5
            return -0.40 + 0.22 * (z - 0.5)^3.0
        elseif z > -0.3
            return -0.50 - 0.06 * sin((z + 0.3) / 0.8 * π)
        else
            return -0.48 + 0.04 * ((z + 0.3) / 0.5)^2
        end
    end
    
    for z_pos in [1.05, 0.95, 0.85, 0.7, 0.55, 0.4]
        y_curve = spine_curve(z_pos)
        draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, y_curve, z_pos, 0.084, 0.084, 0.084; ex=(2.0,2.0,2.0), intensity=spine_intensity)
    end
    for z_pos in [0.25, 0.1, -0.05, -0.2]
        y_curve = spine_curve(z_pos)
        draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, y_curve, z_pos, 0.084, 0.084, 0.084; ex=(2.0,2.0,2.0), intensity=spine_intensity)
    end
    for z_pos in [-0.4, -0.6, -0.8, -1.0]
        y_curve = spine_curve(z_pos)
        draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.0, y_curve, z_pos, 0.0945, 0.0945, 0.0945; ex=(2.0,2.0,2.0), intensity=spine_intensity)
    end
end

"""
Helper function to add ribs.
"""
function add_ribs_axes!(phantom, ax_x, ax_y, ax_z)
    rib_intensity = 0.55
    
    # Helper function to create curved rib with variable arc coverage
    function create_rib_curve(z_pos, num_segments, torso_width, torso_depth, arc_coverage)
        function spine_curve_local(z)
            if z > 0.5
                return -0.45 + 0.18 * (z - 0.5)^2.2
            elseif z > -0.3
                return -0.50 - 0.06 * sin((z + 0.3) / 0.8 * π)
            else
                return -0.48 + 0.04 * ((z + 0.3) / 0.5)^2
            end
        end
        spine_y = spine_curve_local(z_pos)
        
        # arc_coverage: 1.0 = full 360° (complete circle), 0.5 = 180° (posterior only)
        if arc_coverage >= 1.0
            angles = range(-3π/2, π/2, length=num_segments)
        else
            total_angle = 2π * arc_coverage
            angle_center = -π/2  # Posterior (at spine)
            angle_start = angle_center - total_angle/2
            angle_end = angle_center + total_angle/2
            angles = range(angle_start, angle_end, length=Int(round(num_segments * arc_coverage)))
        end
        
        for angle in angles
            # Rib attachment: ribs attach to anterior surface of spine
            spine_radius = (z_pos > -0.3) ? 0.084 : 0.0945
            spine_diameter = 2 * spine_radius
            rib_attachment_y = spine_y + spine_radius - spine_diameter
            
            # Ribs extend from spine (posterior) wrapping to anterior
            x_pos = torso_width * cos(angle)
            y_pos = rib_attachment_y + torso_depth + torso_depth * sin(angle)
            
            # Ribs slope downward anteriorly (higher at posterior/spine, lower anteriorly)
            z_adjustment = (π - abs(π/2 + angle)) / (2π) * 0.06
            
            draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, x_pos, y_pos, z_pos + z_adjustment, 0.04, 0.04, 0.055; ex=(2.0,2.0,2.0), intensity=rib_intensity)
        end
    end
    
    num_segments = 80
    create_rib_curve(0.6, num_segments, 0.64, 0.53, 1.0)
    create_rib_curve(0.45, num_segments, 0.68, 0.58, 1.0)
    create_rib_curve(0.3, num_segments, 0.72, 0.61, 1.0)
    create_rib_curve(0.15, num_segments, 0.76, 0.62, 1.0)
    
    create_rib_curve(0.0, num_segments, 0.80, 0.62, 1.0)
    create_rib_curve(-0.15, num_segments, 0.80, 0.62, 0.9)
    
    create_rib_curve(-0.3, num_segments, 0.78, 0.58, 0.75)
    create_rib_curve(-0.45, num_segments, 0.75, 0.57, 0.6)
    create_rib_curve(-0.6, num_segments, 0.65, 0.53, 0.5)
end

function add_arm_bones_axes!(phantom, ax_x, ax_y, ax_z)
    bone_intensity = 0.55
    
    # Left arm bones
    arm_bone_positions_l = [
        (-0.50, -0.28, 0.58, 0.150, 0.075, 0.150),
        (-0.55, -0.25, 0.56, 0.160, 0.080, 0.200),
        (-0.60, -0.23, 0.54, 0.165, 0.083, 0.260),
        (-0.65, -0.20, 0.52, 0.170, 0.100, 0.350),
        (-0.70, -0.15, 0.51, 0.175, 0.120, 0.220),
        (-0.75, -0.05, 0.50, 0.175, 0.150, 0.175),
        (-0.80, 0.00, 0.50, 0.170, 0.170, 0.170),
        (-0.85, 0.00, 0.50, 0.165, 0.165, 0.165),
        (-0.90, 0.00, 0.48, 0.160, 0.160, 0.160),
        (-0.95, 0.00, 0.45, 0.155, 0.155, 0.155),
        (-1.00, 0.00, 0.42, 0.150, 0.150, 0.150),
        (-1.05, 0.00, 0.39, 0.145, 0.145, 0.145),
        (-1.10, 0.00, 0.36, 0.140, 0.140, 0.140),
        (-1.15, 0.00, 0.33, 0.135, 0.135, 0.135),
        (-1.20, 0.00, 0.30, 0.130, 0.130, 0.130),
    ]
    
    for (x_pos, y_pos, z_pos, radius_x, radius_y, radius_z) in arm_bone_positions_l
        draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, x_pos, y_pos - 0.165, z_pos, radius_x/2, radius_y/2, radius_z/2; ex=(2.0,2.0,2.0), intensity=bone_intensity)
    end
    
    # Right arm bones
    arm_bone_positions_r = [
        (0.50, -0.28, 0.58, 0.150, 0.075, 0.150),
        (0.55, -0.25, 0.56, 0.160, 0.080, 0.200),
        (0.60, -0.23, 0.54, 0.165, 0.083, 0.260),
        (0.65, -0.20, 0.52, 0.170, 0.100, 0.350),
        (0.70, -0.15, 0.51, 0.175, 0.120, 0.220),
        (0.75, -0.05, 0.50, 0.175, 0.150, 0.175),
        (0.80, 0.00, 0.50, 0.170, 0.170, 0.170),
        (0.85, 0.00, 0.50, 0.165, 0.165, 0.165),
        (0.90, 0.00, 0.48, 0.160, 0.160, 0.160),
        (0.95, 0.00, 0.45, 0.155, 0.155, 0.155),
        (1.00, 0.00, 0.42, 0.150, 0.150, 0.150),
        (1.05, 0.00, 0.39, 0.145, 0.145, 0.145),
        (1.10, 0.00, 0.36, 0.140, 0.140, 0.140),
        (1.15, 0.00, 0.33, 0.135, 0.135, 0.135),
        (1.20, 0.00, 0.30, 0.130, 0.130, 0.130),
    ]
    
    for (x_pos, y_pos, z_pos, radius_x, radius_y, radius_z) in arm_bone_positions_r
        draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, x_pos, y_pos - 0.165, z_pos, radius_x/2, radius_y/2, radius_z/2; ex=(2.0,2.0,2.0), intensity=bone_intensity)
    end
end

"""
Helper function to add liver.
"""
function add_liver_axes!(phantom, ax_x, ax_y, ax_z)
    # Liver (right upper abdomen)
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, 0.25, 0.15, -0.80, 0.35, 0.30, 0.30; ex=(2.5,2.5,2.5), intensity=0.45)
    
    # Left lobe
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -0.05, 0.12, -0.75, 0.20, 0.25, 0.25; ex=(2.5,2.5,2.5), intensity=0.43)
end

"""
Helper function to add stomach.
"""
function add_stomach_axes!(phantom, ax_x, ax_y, ax_z)
    # Stomach (left upper abdomen)
    # Fundus
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -0.25, 0.05, -0.70, 0.30, 0.18, 0.20; ex=(2.5,2.5,2.5), intensity=0.38)
    
    # Body
    draw_superellipsoid!(phantom, ax_x, ax_y, ax_z, -0.15, 0.08, -0.80, 0.16, 0.16, 0.22; ex=(2.5,2.5,2.5), intensity=0.36)
end
