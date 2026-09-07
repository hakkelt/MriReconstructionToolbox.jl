"""
	hard_consistency_prox(𝒜, y, maxit, tol)

Build the proximable indicator function of the hard data consistency constraint
``\\{x \\mid \\mathcal{A}x = y\\}``, whose proximal operator is the orthogonal projection
``\\operatorname{proj}(x) = x - \\mathcal{A}^* (\\mathcal{A} \\mathcal{A}^*)^{-1} (\\mathcal{A}x - y)``.

The projection itself has no MRI-specific content, so it lives upstream as
`ProximalOperators.IndAffineCG` (see NAMING.md rule 7.2): a matrix-free `IndAffine` that solves
``(\\mathcal{A} \\mathcal{A}^*) v = r`` by Conjugate Gradient using only `𝒜 * ·` and `𝒜' * ·`, up to
`maxit` iterations and tolerance `tol`.

What is MRI-specific, and therefore stays here, is knowing *when* the closed form applies: when
`is_AAc_diagonal(𝒜)` is true (single-coil Cartesian, or a `KSpaceToImage` signal model),
``(\\mathcal{A} \\mathcal{A}^*)^{-1}`` is a division by `diag_AAc(𝒜)` and no CG iteration is needed.
That diagonal is handed to `IndAffineCG` as its `AAc_diag`.
"""
function hard_consistency_prox(𝒜, y, maxit::Int, tol::Real)
    AAc_diag = is_AAc_diagonal(𝒜) ? diag_AAc(𝒜) : nothing
    return IndAffineCG(𝒜, y; maxit, tol, AAc_diag)
end
