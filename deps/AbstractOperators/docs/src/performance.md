# Performance

For end users, the main performance recommendations are:

1. Prefer `mul!` over `*` to avoid allocations.
2. Preallocate with `allocate_in_domain(op)` / `allocate_in_codomain(op)`.
3. Use operator constructors with array inputs to preserve storage type (CPU vs GPU).

Developer-oriented performance internals (threading heuristics, storage traits, backend caveats) were moved to [Custom Operators](@ref) to keep this page concise.
