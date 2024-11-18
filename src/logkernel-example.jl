using ContinuumArrays, SingularIntegrals, ClassicalOrthogonalPolynomials, Test

# Pointwise Evaluation Using the LogKernel:

# Use the Legendre OPs and consider a point charge at iϵ.
P = Legendre(); x_axis = axes(P, 1); ϵ = 1e-4;

# Pick a function to consider:
f = expand(P, λ -> λ^2);

# Choose a point of evaluation x₀:
x₀ = -1.5;

# Use the logkernel to evaluate ∫₋₁¹ log (x₀ - x²) dx 
@test log.(abs.(x₀ .- x_axis')) * f == logkernel(f, x₀) == logkernel(P, x₀) * coefficients(f);

# Ordered by slowest ⟹ largest
@time log.(abs.(x₀ .- x_axis')) * f;
@time logkernel(P, x₀) * coefficients(f);
@time logkernel(f, x₀);