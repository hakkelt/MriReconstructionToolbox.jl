#=
NotebookUtils.jl

Shared preamble for the docs/notebooks/*.ipynb example notebooks. Every notebook's first code
cell does:

    include("NotebookUtils.jl")
    using .NotebookUtils

This module exists so the 11 notebooks do not each re-fix the same two display glitches, and do
not each re-implement the same three or four tiny plotting helpers. See docs/notebooks/README.md
for how it fits into the notebook workflow.
=#
module NotebookUtils

using MIRTjim: jim, jim!
using Plots: Plot
using LinearAlgebra: norm

export nrmse, side_by_side, difference_image

# --------------------------------------------------------------------------------------------
# 1. jim orientation: MIRTjim's `yflip` default is `minimum(y) >= 0`, which is `true` for the
#    plain pixel-index axes every notebook uses (`y = 1:size(z, 2)`). That default renders the
#    Shepp-Logan / torso phantoms upside down relative to how they are conventionally shown
#    (the standard picture has the off-centre gray "ventricle" ellipse in the UPPER half and the
#    small high-contrast "tumour" markers in the LOWER half — verified visually against this
#    package's own `create_shepp_logan_phantom` output, see the A0 handoff report). Overriding
#    the *global* default here — instead of passing `yflip = false` at every call site — is the
#    single point of control the campaign plan asks for: it fixes every `jim` call in every
#    notebook, present and future, without touching the phantom arrays themselves.
jim(:yflip, false)

# --------------------------------------------------------------------------------------------
# 2. Write plot titles and axis labels in plain ASCII (`title = "Ax"`, not `title = "𝒜x"`).
#    Script letters stay in markdown prose and as Julia variable names.
#
# --------------------------------------------------------------------------------------------
# 3. Shared helpers the notebooks otherwise redefine ad hoc (surveyed across all 11 notebooks
#    before writing these): a one- or two-line `nrmse(x̂) = norm(abs.(x̂) - abs.(x_true)) /
#    norm(abs.(x_true))` closure-over-x_true appears in notebooks 04/05/09 (`nrmse`), 07
#    (`nrmse_dyn`) and 08 (`aligned_nrmse`); side-by-side `jim` panels built by hand from
#    `p1 = jim(...); p2 = jim(...); jim(p1, p2; layout = (1, 2))` appear throughout without a
#    shared color scale; and a `title = "Error"` difference panel (`jim(abs.(x̂ - x); title =
#    "Error")`) appears repeatedly (notebook 01 and others). These three replace those.

"""
    nrmse(x̂, x) -> Real

Normalized RMSE between a reconstruction `x̂` and a reference `x`, magnitude-only (so it is
meaningful for complex images):

    nrmse(x̂, x) = ‖|x̂| - |x|‖₂ / ‖|x|‖₂

This is the two-argument, referentially-transparent form; a notebook that repeatedly compares
against the same ground truth may still define a local one-argument closure
`nrmse1(x̂) = nrmse(x̂, x_true)` for brevity.
"""
nrmse(x̂, x) = norm(abs.(x̂) .- abs.(x)) / norm(abs.(x))

"""
    side_by_side(images...; titles = ("", "", ...), clim = nothing, size = (350*n, 350), kwargs...)

Display several images side by side (via `jim`/`MIRTjim`) on a SHARED color scale, so the panels
are visually comparable. By default `clim` is the joint `(min, max)` of `abs.(image)` across all
`images`; pass `clim` explicitly to override. `titles` pairs with `images` positionally.
Remaining `kwargs` are forwarded to every panel's `jim` call (not to the combining `jim`).
"""
function side_by_side(
        images...;
        titles = ntuple(_ -> "", length(images)),
        clim = nothing,
        size = (350 * length(images), 350),
        kwargs...,
    )
    length(titles) == length(images) ||
        throw(ArgumentError("side_by_side: got $(length(images)) images but $(length(titles)) titles"))
    mags = map(img -> abs.(img), images)
    shared_clim = clim === nothing ? (minimum(minimum, mags), maximum(maximum, mags)) : clim
    panels = [
        jim(img; title = t, clim = shared_clim, kwargs...)
            for (img, t) in zip(images, titles)
    ]
    return jim(panels...; layout = (1, length(images)), size = size)
end

"""
    difference_image(x̂, x; title = "Error", scale = 1, kwargs...)

Display the magnitude difference image `abs.(x̂ .- x)` with ITS OWN color scale (errors are
typically much smaller in magnitude than the images being compared, so sharing `clim` with them
would make the error panel look uniformly black). `scale` multiplies the difference before
display, e.g. to make a small error visible; the title records the scale factor when `scale != 1`.
`kwargs` are forwarded to `jim`.
"""
function difference_image(x̂, x; title = "Error", scale = 1, kwargs...)
    d = scale .* abs.(x̂ .- x)
    displayed_title = scale == 1 ? title : "$title (×$scale)"
    return jim(d; title = displayed_title, kwargs...)
end

end # module
