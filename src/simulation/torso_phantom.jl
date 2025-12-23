"""
Helper function to add torso outer boundary with shoulders, neck, and arms.
Uses smaller superellipsoids (n=2.5) with merged segments for efficiency.
"""
function add_torso_boundary!(phantom, X_norm, Y_norm, Z_norm)
    # Neck - 20% thicker, moved 10% more posteriorly
    # Original: radius 0.28/0.26, Y_offset -0.15
    # New: radius 0.336/0.312 (20% thicker), Y_offset -0.165 (10% more posterior)
    neck_lower = (abs.(X_norm) ./ 0.312).^2.5 .+ (abs.((Y_norm .+ 0.165)) ./ 0.336).^2.5 .+ (abs.(Z_norm .- 0.75) ./ 0.22).^2.5 .<= 1
    phantom[neck_lower] .= 0.2
    
    neck_upper = (abs.(X_norm) ./ 0.312).^2.5 .+ (abs.((Y_norm .+ 0.165)) ./ 0.312).^2.5 .+ (abs.(Z_norm .- 0.92) ./ 0.15).^2.5 .<= 1
    phantom[neck_upper] .= 0.2
    
    # Shoulders - connection point for arms
    shoulder_l = (abs.(X_norm) ./ 0.8).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.28).^2.5 .+ (abs.(Z_norm .- 0.70) ./ 0.25).^2.5 .<= 1
    phantom[shoulder_l] .= 0.2
    
    # Arms - larger, moved higher to align with shoulders, extending outside image bounds
    # Left arm - upper (at shoulder level)
    arm_l_upper = (abs.(X_norm .+ 0.68) ./ 0.18).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.18).^2.5 .+ (abs.(Z_norm .- 0.62) ./ 0.28).^2.5 .<= 1
    phantom[arm_l_upper] .= 0.2
    
    # Left arm - mid (extending further)
    arm_l_mid = (abs.(X_norm .+ 0.88) ./ 0.17).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.17).^2.5 .+ (abs.(Z_norm .- 0.50) ./ 0.26).^2.5 .<= 1
    phantom[arm_l_mid] .= 0.2
    
    # Left arm - lower (reaching outside bounds)
    arm_l_lower = (abs.(X_norm .+ 1.05) ./ 0.16).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.16).^2.5 .+ (abs.(Z_norm .- 0.38) ./ 0.24).^2.5 .<= 1
    phantom[arm_l_lower] .= 0.2
    
    # Right arm - upper (at shoulder level)
    arm_r_upper = (abs.(X_norm .- 0.68) ./ 0.18).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.18).^2.5 .+ (abs.(Z_norm .- 0.62) ./ 0.28).^2.5 .<= 1
    phantom[arm_r_upper] .= 0.2
    
    # Right arm - mid (extending further)
    arm_r_mid = (abs.(X_norm .- 0.88) ./ 0.17).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.17).^2.5 .+ (abs.(Z_norm .- 0.50) ./ 0.26).^2.5 .<= 1
    phantom[arm_r_mid] .= 0.2
    
    # Right arm - lower (reaching outside bounds)
    arm_r_lower = (abs.(X_norm .- 1.05) ./ 0.16).^2.5 .+ (abs.(Y_norm .+ 0.165) ./ 0.16).^2.5 .+ (abs.(Z_norm .- 0.38) ./ 0.24).^2.5 .<= 1
    phantom[arm_r_lower] .= 0.2
    
    # Chest - merged rib segments (3 segments for ribs 1-12)
    # Ribs 1-4 level (upper chest)
    chest_upper = (abs.(X_norm) ./ 0.86).^2.5 .+ (abs.(Y_norm) ./ 0.69).^2.5 .+ (abs.(Z_norm .- 0.45) ./ 0.35).^2.5 .<= 1
    phantom[chest_upper] .= 0.2
    
    # Ribs 5-8 level (mid chest - widest)
    chest_mid = (abs.(X_norm) ./ 0.93).^2.5 .+ (abs.(Y_norm) ./ 0.72).^2.5 .+ (abs.(Z_norm .- 0.17) ./ 0.32).^3.5 .<= 1
    phantom[chest_mid] .= 0.2
    
    # Ribs 9-12 level (lower chest, transitioning to abdomen)
    chest_lower = (abs.(X_norm) ./ 0.91).^2.5 .+ (abs.(Y_norm) ./ 0.71).^2.5 .+ (abs.(Z_norm .+ 0.11) ./ 0.32).^3.5 .<= 1
    phantom[chest_lower] .= 0.2
    
    # Abdomen - merged into 2 segments (was 6)
    # Upper abdomen (waist region)
    abdomen_upper = (abs.(X_norm) ./ 0.87).^2.5 .+ (abs.(Y_norm) ./ 0.67).^2.5 .+ (abs.(Z_norm .+ 0.39) ./ 0.32).^3.5 .<= 1
    phantom[abdomen_upper] .= 0.2
    
    # Lower abdomen (narrower, pelvis is out of image inferiorly)
    abdomen_lower = (abs.(X_norm) ./ 0.83).^2.5 .+ (abs.(Y_norm) ./ 0.62).^2.5 .+ (abs.(Z_norm .+ 0.85) ./ 0.45).^3.5 .<= 1
    phantom[abdomen_lower] .= 0.2
    
    # Posterior extensions for spine/back coverage - three superellipsoids to follow torso shape
    # Increased thickness toward anterior direction (Y+ axis is anterior)
    # Upper back (cervical/upper thoracic region) - more anteriorly extended
    back_upper = (abs.(X_norm) ./ 0.70).^2.5 .+ (abs.(Y_norm .+ 0.28) ./ 0.43).^2.5 .+ (abs.(Z_norm .- 0.35) ./ 0.50).^2.5 .<= 1
    phantom[back_upper] .= 0.2
    
    # Mid back (mid thoracic region) - more anteriorly extended
    back_mid = (abs.(X_norm) ./ 0.75).^2.5 .+ (abs.(Y_norm .+ 0.28) ./ 0.47).^2.5 .+ (abs.(Z_norm .+ 0.10) ./ 0.55).^2.5 .<= 1
    phantom[back_mid] .= 0.2
    
    # Lower back (lumbar region) - more anteriorly extended
    back_lower = (abs.(X_norm) ./ 0.78).^2.5 .+ (abs.(Y_norm .+ 0.15) ./ 0.48).^2.5 .+ (abs.(Z_norm .+ 0.60) ./ 0.55).^2.5 .<= 1
    phantom[back_lower] .= 0.2
    
    # Shoulder and arm bones moved to separate function, called earlier
end

"""
Helper function to add shoulder and arm bones.
Series of ellipsoids from spine to image edge.
"""
function add_arm_bones!(phantom, X_norm, Y_norm, Z_norm)
    bone_intensity = 0.55  # Lower than heart (0.6)
    
    # Left arm bones - series of ellipsoids from spine region to lateral edge
    # Starting near spine at Y=-0.5, extending to X=-1.2 (beyond image edge)
    # Shoulder arc (first 6 bones) has half thickness in y-direction
    arm_bone_positions_l = [
        (-0.50, -0.37, 0.58, 0.150, 0.075),  # Near spine, superior - half y-radius
        (-0.55, -0.35, 0.56, 0.160, 0.080),
        (-0.60, -0.33, 0.54, 0.165, 0.083),
        (-0.65, -0.30, 0.52, 0.170, 0.100),  # Scapula region
        (-0.70, -0.25, 0.51, 0.175, 0.120),
        (-0.75, -0.15, 0.50, 0.175, 0.150),  # Transitioning to arm
        (-0.80, -0.05, 0.50, 0.170, 0.170),  # Regular arm bones from here
        (-0.85, 0.00, 0.50, 0.165, 0.165),
        (-0.90, 0.00, 0.48, 0.160, 0.160),   # Mid arm
        (-0.95, 0.00, 0.45, 0.155, 0.155),
        (-1.00, 0.00, 0.42, 0.150, 0.150),
        (-1.05, 0.00, 0.39, 0.145, 0.145),
        (-1.10, 0.00, 0.36, 0.140, 0.140),
        (-1.15, 0.00, 0.33, 0.135, 0.135),   # Near edge
        (-1.20, 0.00, 0.30, 0.130, 0.130),   # Beyond edge
    ]
    
    for (x_pos, y_pos, z_pos, radius_x, radius_y) in arm_bone_positions_l
        bone = (abs.(X_norm .- x_pos) ./ (radius_x / 2.0)).^2 .+ 
               (abs.(Y_norm .- y_pos .+ 0.165) ./ (radius_y / 2.0)).^2 .+ 
               (abs.(Z_norm .- z_pos) ./ (radius_x / 2.0)).^2 .<= 1
        phantom[bone] .= bone_intensity
    end
    
    # Right arm bones - mirror of left
    arm_bone_positions_r = [
        (0.50, -0.37, 0.58, 0.150, 0.075),
        (0.55, -0.35, 0.56, 0.160, 0.080),
        (0.60, -0.33, 0.54, 0.165, 0.083),
        (0.65, -0.30, 0.52, 0.170, 0.100),
        (0.70, -0.25, 0.51, 0.175, 0.120),
        (0.75, -0.15, 0.50, 0.175, 0.150),
        (0.80, -0.05, 0.50, 0.170, 0.170),
        (0.85, 0.00, 0.50, 0.165, 0.165),
        (0.90, 0.00, 0.48, 0.160, 0.160),
        (0.95, 0.00, 0.45, 0.155, 0.155),
        (1.00, 0.00, 0.42, 0.150, 0.150),
        (1.05, 0.00, 0.39, 0.145, 0.145),
        (1.10, 0.00, 0.36, 0.140, 0.140),
        (1.15, 0.00, 0.33, 0.135, 0.135),
        (1.20, 0.00, 0.30, 0.130, 0.130),
    ]
    
    for (x_pos, y_pos, z_pos, radius_x, radius_y) in arm_bone_positions_r
        bone = (abs.(X_norm .- x_pos) ./ (radius_x / 2.0)).^2 .+ 
               (abs.(Y_norm .- y_pos .+ 0.165) ./ (radius_y / 2.0)).^2 .+ 
               (abs.(Z_norm .- z_pos) ./ (radius_x / 2.0)).^2 .<= 1
        phantom[bone] .= bone_intensity
    end
end

"""
Helper function to add lungs with 4 components each:
1. Upper lobe (top of lung)
2. Lower lobe (bottom of lung)
3. Heart cavity
4. Diaphragm
"""
function add_lungs!(phantom, X_norm, Y_norm, Z_norm)
    # Reference rib parameters for lung sizing
    # Rib 3 at z=0.3: width=0.74, depth=0.62
    # Rib 4 at z=0.15: width=0.78, depth=0.62
    # Rib 5 at z=0.0: width=0.80, depth=0.62
    # Rib 9 at z=-0.6: width=0.65, depth=0.53
    
    # Lungs moved 5% closer to each other (x-offset reduced by 5% of 2.0 = 0.1)
    # Additional 5% closer = 0.1 more (total 0.2 reduction)
    # Original: ±0.42, Now: ±0.32
    lung_x_offset = 0.32
    
    # Left Lung - 4 component structure
    # 1. Top of lung: superellipsoid with nx,ny=2.5, nz=1.5
    #    Center aligned with rib 3 (z=0.3), stretched 5% more toward head and 20% more toward bottom
    #    Top ellipsoid closer to each other by 15% = 0.3 total reduction in x-offset
    lung_l_top_x = lung_x_offset - 0.15  # 15% closer = 0.15 reduction
    lung_l_top_radius = 0.74 * 0.48  # Slightly less than semi-minor axis of rib 3
    # Original height: 0.35, increase by 5% up and 20% down: new center shifted down, new height increased
    # 5% toward head = 0.35 * 0.05 = 0.0175, 20% toward bottom = 0.35 * 0.20 = 0.07
    # New center: 0.315 - (0.07 - 0.0175)/2 = 0.315 - 0.02625 = 0.289
    # New height: 0.35 + 0.0175 + 0.07 = 0.4375
    lung_l_top = (abs.(X_norm .+ lung_l_top_x) ./ lung_l_top_radius).^2.5 .+ 
                 (abs.(Y_norm) ./ lung_l_top_radius).^2.5 .+ 
                 (abs.(Z_norm .- 0.289) ./ 0.4375).^1.5 .<= 1
    phantom[lung_l_top] .= 0.08
    
    # 2. Lower lobe: superellipsoid with n=2.5 in all directions
    #    Center between rib 5 and rib 9, bottom slightly lower than rib 9 (z=-0.65)
    #    Size increased by 5% in x-y (bottom of lung)
    lung_l_lower_radius = 0.80 * 0.48 * 1.05  # 5% larger
    lung_l_lower = (abs.(X_norm .+ lung_x_offset) ./ lung_l_lower_radius).^2.5 .+ 
                   (abs.(Y_norm) ./ lung_l_lower_radius).^2.5 .+ 
                   (abs.(Z_norm .+ 0.20) ./ 0.45).^2.5 .<= 1  # Extended lower
    phantom[lung_l_lower] .= 0.09
    
    # 3. Diaphragm: decreased in x and y, moved lower relative to lower lobes
    #    Decreased by 20% in x-y
    diaphragm_radius = lung_l_lower_radius * 0.80  # 20% smaller
    diaphragm_l = (abs.(X_norm .+ lung_x_offset) ./ diaphragm_radius).^2.5 .+ 
                  (abs.(Y_norm) ./ diaphragm_radius).^2.5 .+ 
                  (abs.(Z_norm .+ 0.70) ./ 0.20).^1.5 .<= 1  # Moved lower
    phantom[diaphragm_l] .= 0.2
    
    # Right Lung - 4 component structure (mirror of left)
    # 1. Top of lung
    lung_r_top_x = lung_x_offset - 0.15  # 15% closer
    lung_r_top = (abs.(X_norm .- lung_r_top_x) ./ lung_l_top_radius).^2.5 .+ 
                 (abs.(Y_norm) ./ lung_l_top_radius).^2.5 .+ 
                 (abs.(Z_norm .- 0.289) ./ 0.4375).^1.5 .<= 1
    phantom[lung_r_top] .= 0.08
    
    # 2. Lower lobe
    lung_r_lower = (abs.(X_norm .- lung_x_offset) ./ lung_l_lower_radius).^2.5 .+ 
                   (abs.(Y_norm) ./ lung_l_lower_radius).^2.5 .+ 
                   (abs.(Z_norm .+ 0.20) ./ 0.45).^2.5 .<= 1
    phantom[lung_r_lower] .= 0.09
    
    # 4. Diaphragm (right)
    diaphragm_r = (abs.(X_norm .- lung_x_offset) ./ diaphragm_radius).^2.5 .+ 
                  (abs.(Y_norm) ./ diaphragm_radius).^2.5 .+ 
                  (abs.(Z_norm .+ 0.70) ./ 0.20).^1.5 .<= 1
    phantom[diaphragm_r] .= 0.2
    
    # +1 Heart cavity: single ellipsoid at heart center (X≈0.05, Y≈0, Z≈0.05)
    heart_cavity = (abs.(X_norm .- 0.05) ./ 0.30).^2.2 .+ 
                   (abs.(Y_norm) ./ 0.30).^2.2 .+ 
                   (abs.(Z_norm .+ 0.00) ./ 0.50).^1.5 .<= 1
    phantom[heart_cavity] .= 0.2
end

"""
Helper function to add heart chambers with realistic conical shape.
Uses varying superellipsoid exponents to create apex at bottom.
"""
function add_heart!(phantom, X_norm, Y_norm, Z_norm)
    # Heart - Main body with varying exponents for conical shape
    # Upper part (base): broader, lower exponent
    # Lower part (apex): narrower, higher exponent for sharper cone
    
    # Upper heart (base) - superellipsoid n=2.0 (rounder)
    heart_upper = (abs.(X_norm .- 0.05) ./ 0.25).^2.0 .+ 
                  (abs.(Y_norm) ./ 0.25).^2.0 .+ 
                  (abs.((Z_norm .+ 0.1)) ./ 0.20).^2.0 .<= 1
    phantom[heart_upper] .= 0.6
    
    # Mid heart - transitional shape with n=2.5
    heart_mid = (abs.(X_norm .- 0.05) ./ 0.23).^2.5 .+ 
                (abs.(Y_norm) ./ 0.23).^2.5 .+ 
                (abs.(Z_norm) ./ 0.18).^2.5 .<= 1
    phantom[heart_mid] .= 0.6
    
    # Lower heart (apex) - conical with varying exponents
    # X and Y have higher exponent (sharper), Z has lower (elongated)
    heart_apex = (abs.(X_norm .- 0.05) ./ 0.18).^3.5 .+ 
                 (abs.(Y_norm) ./ 0.18).^3.5 .+ 
                 (abs.((Z_norm .- 0.15)) ./ 0.22).^2.0 .<= 1
    phantom[heart_apex] .= 0.6
    
    # Left ventricle (larger chamber) - conical shape
    heart_lv_upper = (abs.(X_norm .- 0.0) ./ 0.18).^2.0 .+ 
                     (abs.((Y_norm .- 0.02)) ./ 0.18).^2.0 .+ 
                     (abs.((Z_norm .- 0.00)) ./ 0.15).^2.0 .<= 1
    phantom[heart_lv_upper] .= 0.7
    
    heart_lv_apex = (abs.(X_norm .- 0.0) ./ 0.14).^3.0 .+ 
                    (abs.((Y_norm .- 0.02)) ./ 0.14).^3.0 .+ 
                    (abs.((Z_norm .- 0.12)) ./ 0.15).^2.0 .<= 1
    phantom[heart_lv_apex] .= 0.7
    
    # Right ventricle - also conical
    heart_rv_upper = (abs.(X_norm .- 0.10) ./ 0.15).^2.0 .+ 
                     (abs.(Y_norm) ./ 0.15).^2.0 .+ 
                     (abs.((Z_norm .- 0.00)) ./ 0.14).^2.0 .<= 1
    phantom[heart_rv_upper] .= 0.65
    
    heart_rv_apex = (abs.(X_norm .- 0.10) ./ 0.12).^3.0 .+ 
                    (abs.(Y_norm) ./ 0.12).^3.0 .+ 
                    (abs.((Z_norm .- 0.12)) ./ 0.13).^2.0 .<= 1
    phantom[heart_rv_apex] .= 0.65
    
    # Left atrium (at base of heart)
    heart_la = (abs.(X_norm .- 0.03) ./ 0.12).^2.2 .+ 
               (abs.((Y_norm .+ 0.05)) ./ 0.12).^2.2 .+ 
               (abs.((Z_norm .+ 0.2)) ./ 0.15).^2.2 .<= 1
    phantom[heart_la] .= 0.68
    
    # Right atrium (at base of heart)
    heart_ra = (abs.(X_norm .- 0.03) ./ 0.12).^2.2 .+ 
               (abs.((Y_norm .+ 0.07)) ./ 0.12).^2.2 .+ 
               (abs.((Z_norm .+ 0.2)) ./ 0.15).^2.2 .<= 1
    phantom[heart_ra] .= 0.63
end

"""
Helper function to add major vessels as superellipsoid segments (not tubes).
"""
function add_vessels!(phantom, X_norm, Y_norm, Z_norm)
    # Helper function to create vessel segment as superellipsoid
    function create_vessel_segment!(phantom, X_norm, Y_norm, Z_norm, x_center, y_center, z_center, radius_xy, height_z, intensity, n=2.5)
        vessel_mask = (abs.(X_norm .- x_center) ./ radius_xy).^n .+ 
                      (abs.(Y_norm .- y_center) ./ radius_xy).^n .+ 
                      (abs.(Z_norm .- z_center) ./ height_z).^n .<= 1
        phantom[vessel_mask] .= intensity
    end
    
    # Aorta (ascending) - vertical vessel made of superellipsoid segments
    # Center: X=-0.02, Y=-0.05, Z from 0.1 to 0.7 (superior direction)
    aorta_segments = [
        (0.70, 0.08),  # top
        (0.55, 0.12),
        (0.40, 0.12),
        (0.25, 0.12),
        (0.10, 0.12)   # bottom
    ]
    for (z_center, half_height) in aorta_segments
        create_vessel_segment!(phantom, X_norm, Y_norm, Z_norm, -0.02, -0.05, z_center, 0.06, half_height, 0.8)
    end
    
    # Pulmonary artery - vertical vessel
    # Center: X=-0.05, Y=-0.05, Z from 0.05 to 0.6
    pa_segments = [
        (0.60, 0.08),
        (0.47, 0.10),
        (0.32, 0.12),
        (0.17, 0.12),
        (0.05, 0.10)
    ]
    for (z_center, half_height) in pa_segments
        create_vessel_segment!(phantom, X_norm, Y_norm, Z_norm, -0.05, -0.05, z_center, 0.05, half_height, 0.75)
    end
    
    # Superior vena cava - vertical vessel
    # Center: X=0.1, Y=-0.05, Z from 0.35 to 0.85 (from heart to superior)
    svc_segments = [
        (0.85, 0.08),
        (0.72, 0.10),
        (0.58, 0.12),
        (0.45, 0.12),
        (0.35, 0.08)
    ]
    for (z_center, half_height) in svc_segments
        create_vessel_segment!(phantom, X_norm, Y_norm, Z_norm, 0.1, -0.05, z_center, 0.04, half_height, 0.78)
    end
end

"""
Helper function to add spine (vertebral column) with realistic spinal curvature.
Includes cervical lordosis, thoracic kyphosis, and lumbar lordosis.
Extended by one vertebra at top and bottom.
"""
function add_spine!(phantom, X_norm, Y_norm, Z_norm)
    spine_intensity = 0.55  # Lower than heart (0.6)
    
    # Spinal curvature function: Y offset based on Z position
    # Creates realistic S-curve (cervical lordosis, thoracic kyphosis, lumbar lordosis)
    # Extended to cover C0/Atlas at top (z=1.05) and L6/sacral at bottom (z=-1.0)
    # Top of spine moved 10% closer to center: Y=-0.50 -> Y=-0.40 (additional 10%)
    function spine_curve(z)
        if z > 0.5
            # Cervical lordosis (forward curve) - even more pronounced, 10% closer to center
            return -0.40 + 0.22 * (z - 0.5)^3.0  # Increased curvature by neck
        elseif z > -0.3
            # Thoracic kyphosis (backward curve) 
            return -0.50 - 0.06 * sin((z + 0.3) / 0.8 * π)
        else
            # Lumbar lordosis (forward curve)
            return -0.48 + 0.04 * ((z + 0.3) / 0.5)^2
        end
    end
    
    # C0/C1/Atlas at top (extended top by one vertebra: z=1.05)
    # Added z=1.05 for additional vertebra
    for z_pos in [1.05, 0.95, 0.85, 0.7, 0.55, 0.4]
        y_curve = spine_curve(z_pos)
        vertebra = (X_norm ./ 0.084).^2 .+ ((Y_norm .- y_curve) ./ 0.084).^2 .+ ((Z_norm .- z_pos) ./ 0.084).^2 .<= 1
        phantom[vertebra] .= spine_intensity
    end
    
    # Mid thoracic vertebrae (T5-T12)
    for z_pos in [0.25, 0.1, -0.05, -0.2]
        y_curve = spine_curve(z_pos)
        vertebra = (X_norm ./ 0.084).^2 .+ ((Y_norm .- y_curve) ./ 0.084).^2 .+ ((Z_norm .- z_pos) ./ 0.084).^2 .<= 1
        phantom[vertebra] .= spine_intensity
    end
    
    # Lower lumbar vertebrae (L1-L6, extended by one at bottom)
    # Added z=-1.0 for additional vertebra at bottom
    for z_pos in [-0.4, -0.6, -0.8, -1.0]
        y_curve = spine_curve(z_pos)
        vertebra = (X_norm ./ 0.0945).^2 .+ ((Y_norm .- y_curve) ./ 0.0945).^2 .+ ((Z_norm .- z_pos) ./ 0.0945).^2 .<= 1
        phantom[vertebra] .= spine_intensity
    end
end

"""
Helper function to add ribs with variable arc coverage.
"""
function add_ribs!(phantom, X_norm, Y_norm, Z_norm)
    rib_intensity = 0.55  # Same as other bones
    
    # Helper function to create curved rib with variable arc coverage
    function create_rib_curve(z_pos, num_segments, torso_width, torso_depth, arc_coverage)
        # Get spine position at this Z level for anatomical connection
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
            # Full rib - complete 360° around the body, centered at spine (-π/2)
            angles = range(-3π/2, π/2, length=num_segments)
        else
            # Partial rib - only posterior portion, centered at spine (-π/2)
            total_angle = 2π * arc_coverage
            angle_center = -π/2  # Posterior (at spine)
            angle_start = angle_center - total_angle/2
            angle_end = angle_center + total_angle/2
            angles = range(angle_start, angle_end, length=Int(round(num_segments * arc_coverage)))
        end
        
        for (i, angle) in enumerate(angles)
            # Rib attachment: ribs attach to anterior surface of spine
            # Move ribs posterior by spine diameter (2 * radius)
            spine_radius = (z_pos > -0.3) ? 0.084 : 0.0945  # Larger in lumbar region
            spine_diameter = 2 * spine_radius
            rib_attachment_y = spine_y + spine_radius - spine_diameter  # Move back by diameter
            
            # Ribs extend from spine (posterior) wrapping to anterior
            # Shift entire ellipse by semi-minor axis in anterior direction
            x_pos = torso_width * cos(angle)
            # Start from spine and extend forward (positive Y direction)
            y_pos = rib_attachment_y + torso_depth + torso_depth * sin(angle)
            
            # Ribs slope downward anteriorly (higher at posterior/spine, lower anteriorly)
            # angle = -π/2 is posterior (spine), angle = π/2 is anterior
            # Distance from posterior: larger when more anterior
            z_adjustment = (π - abs(π/2 + angle)) / (2π) * 0.06
            
            # Create rib segment
            rib_segment = ((X_norm .- x_pos) ./ 0.04).^2 .+ 
                          ((Y_norm .- y_pos) ./ 0.04).^2 .+ 
                          ((Z_norm .- (z_pos + z_adjustment)) ./ 0.055).^2 .<= 1
            phantom[rib_segment] .= rib_intensity
        end
    end
    
    # Heart is approximately centered at Z = 0, with bottom around Z = -0.15
    # Ribs 1-4 above heart (full 360° coverage) - curved ribcage, decreasing diameter toward top
    num_segments = 120
    # Rib 1 (top): 20% smaller diameter (was 10%), forming more curved ribcage
    create_rib_curve(0.6, num_segments, 0.64, 0.53, 1.0)
    # Rib 2: converging toward rib 5 diameter
    create_rib_curve(0.45, num_segments, 0.68, 0.58, 1.0)
    # Rib 3: converging toward rib 5 diameter
    create_rib_curve(0.3, num_segments, 0.72, 0.61, 1.0)
    # Rib 4: nearly matching rib 5 diameter
    create_rib_curve(0.15, num_segments, 0.76, 0.62, 1.0)
    
    # Rib 5 at heart level (full coverage) - reference diameter
    create_rib_curve(0.0, num_segments, 0.80, 0.62, 1.0)    # Full 360° coverage
    create_rib_curve(-0.15, num_segments, 0.80, 0.62, 0.9)  # 90% coverage (~324°)
    
    # Ribs below heart (decreasing arc coverage: 360° → 180°)
    create_rib_curve(-0.3, num_segments, 0.78, 0.58, 0.75)   # 75% coverage (~270°)
    create_rib_curve(-0.45, num_segments, 0.75, 0.57, 0.6)   # 60% coverage (~216°)
    create_rib_curve(-0.6, num_segments, 0.65, 0.53, 0.5)    # 50% coverage (180° - posterior only)
end

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
    # Create ImageGeom for coordinate system
    dims = (nx, ny, nz)
    ig = ImageGeom( ; dims, deltas = fovs ./ dims )
    
    # Get normalized coordinates from ImageGeom
    ax_x = axes(ig)[1]
    ax_y = axes(ig)[2]
    ax_z = axes(ig)[3]
    
    # Create 3D grid
    X = [x for x in ax_x, y in ax_y, z in ax_z]
    Y = [y for x in ax_x, y in ax_y, z in ax_z]
    Z = [z for x in ax_x, y in ax_y, z in ax_z]
    
    # Normalize to [-1, 1] range for easier ellipsoid definitions
    x_range = maximum(ax_x) - minimum(ax_x)
    y_range = maximum(ax_y) - minimum(ax_y)
    z_range = maximum(ax_z) - minimum(ax_z)
    
    X_norm = 2 .* X ./ x_range
    Y_norm = 2 .* Y ./ y_range
    Z_norm = 2 .* Z ./ z_range
    
    # Initialize phantom
    phantom = zeros(Float32, nx, ny, nz)
    
    # Add anatomical structures in order (background to foreground)
    add_torso_boundary!(phantom, X_norm, Y_norm, Z_norm)
    add_arm_bones!(phantom, X_norm, Y_norm, Z_norm)  # Call before ribs
    add_lungs!(phantom, X_norm, Y_norm, Z_norm)
    add_heart!(phantom, X_norm, Y_norm, Z_norm)
    add_vessels!(phantom, X_norm, Y_norm, Z_norm)
    add_spine!(phantom, X_norm, Y_norm, Z_norm)
    add_ribs!(phantom, X_norm, Y_norm, Z_norm)
    add_liver!(phantom, X_norm, Y_norm, Z_norm)
    add_stomach!(phantom, X_norm, Y_norm, Z_norm)
    
    return ComplexF32.(phantom)
end

"""
Helper function to add liver.
"""
function add_liver!(phantom, X_norm, Y_norm, Z_norm)
    # Liver: large organ in right upper abdomen, below diaphragm
    # Positioned on the right side (positive X), anterior (positive Y), below ribs 9-12
    # Aligned with diaphragm at z=-0.70
    # Main liver body - superellipsoid
    liver_main = (abs.(X_norm .- 0.25) ./ 0.35).^2.5 .+ 
                 (abs.(Y_norm .- 0.15) ./ 0.30).^2.5 .+ 
                 (abs.(Z_norm .+ 0.80) ./ 0.30).^2.5 .<= 1
    phantom[liver_main] .= 0.45
    
    # Left lobe of liver (smaller, extends to left side)
    liver_left = (abs.(X_norm .+ 0.05) ./ 0.20).^2.5 .+ 
                 (abs.(Y_norm .- 0.12) ./ 0.25).^2.5 .+ 
                 (abs.(Z_norm .+ 0.75) ./ 0.25).^2.5 .<= 1
    phantom[liver_left] .= 0.43
end

"""
Helper function to add stomach.
"""
function add_stomach!(phantom, X_norm, Y_norm, Z_norm)
    # Stomach: J-shaped organ in left upper abdomen, below diaphragm
    # Positioned on the left side (negative X), anterior to spine
    # Aligned with diaphragm at z=-0.70
    # Fundus (upper part)
    stomach_fundus = (abs.(X_norm .+ 0.25) ./ 0.30).^2.5 .+ 
                     (abs.(Y_norm .- 0.05) ./ 0.18).^2.5 .+ 
                     (abs.(Z_norm .+ 0.70) ./ 0.20).^2.5 .<= 1
    phantom[stomach_fundus] .= 0.38
    
    # Body of stomach
    stomach_body = (abs.(X_norm .+ 0.15) ./ 0.16).^2.5 .+ 
                   (abs.(Y_norm .- 0.08) ./ 0.16).^2.5 .+ 
                   (abs.(Z_norm .+ 0.80) ./ 0.22).^2.5 .<= 1
    phantom[stomach_body] .= 0.36
end
