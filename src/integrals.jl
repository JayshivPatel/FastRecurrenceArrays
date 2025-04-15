import ClassicalOrthogonalPolynomials:
    Legendre,
    OrthogonalPolynomial,
    Ultraspherical,
    orthogonalityweight,
    recurrencecoefficients,
    _p0
import SingularIntegrals: stieltjes

export FastStieltjes, FastLogKernel

function FastStieltjes(n::Integer, x::AbstractVector, f::AbstractVector, P::OrthogonalPolynomial=Legendre())
    w = orthogonalityweight(P)
    A, B, C = recurrencecoefficients(P)
    A, B, C = A[1:n], B[1:n], C[1:n]

    p₀ =  stieltjes(w, x) .* _p0(P)
    p₁ = (A[1] .* x .+ B[1]) .* p₀ .- (A[1]sum(w) * _p0(P))

    return FixedClenshaw(f, A, B, C, x, p₀, p₁)
end

zlog(z) = ifelse(iszero(z), z, z * log(z))
function FastLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    A, B, C = recurrencecoefficients(Ultraspherical(-1/2))
    A, B, C = A[2:n], B[2:n], C[2:n]

    p₀ = @. real(zlog(1 + x) - zlog(x - 1) - 2one(eltype(x)))
    p₁ = @. real((x + 1) * p₀/2 + 1 - zlog(x + 1))
    p₂ = @. real(x * p₁ + 2one(eltype(x))/3)

    return ForwardInplace(f, (A, B, C), real(x), [p₀'; p₁'; p₂'])
end