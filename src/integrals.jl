import ClassicalOrthogonalPolynomials: Legendre, OrthogonalPolynomial, Ultraspherical,
    orthogonalityweight, recurrencecoefficients, _p0
import LinearAlgebra: dot
import SingularIntegrals: stieltjes

export FixedStieltjes, InplaceStieltjes, ThreadedInplaceStieltjes, GPUInplaceStieltjes,
    FixedLogKernel, InplaceLogKernel, ThreadedInplaceLogKernel, GPUInplaceLogKernel

# Forward

function FixedStieltjes(n::Integer, x::AbstractVector, f::AbstractVector, P::OrthogonalPolynomial=Legendre())
    rec_P, input_data = stieltjes_init(n, x, P)
    return dot(FixedRecurrenceArray(real(x), rec_P, n - 2, input_data), f)
end

# Inplace

function InplaceStieltjes(n::Integer, x::AbstractVector, f::AbstractVector, P::OrthogonalPolynomial=Legendre())
    rec_P, input_data = stieltjes_init(n, x, P)
    return ForwardInplace(f, rec_P, real(x), input_data)
end

function ThreadedInplaceStieltjes(n::Integer, x::AbstractVector, f::AbstractVector, P::OrthogonalPolynomial=Legendre())
    rec_P, input_data = stieltjes_init(n, x, P)
    return ThreadedInplace(f, rec_P, real(x), input_data)
end

function GPUInplaceStieltjes(n::Integer, x::AbstractVector, f::AbstractVector, P::OrthogonalPolynomial=Legendre())
    (A, B, C), input_data = stieltjes_init(n, x, P)
    
    # enforce Float32 recurrence coefficients on the GPU
    A, B, C = Float32.(A), Float32.(B), Float32.(C)
    return GPUInplace(f, (A, B, C), real(x), input_data)
end

# Forward 

function FixedLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return dot(FixedRecurrenceArray(real(x), rec_P, n - 2, input_data), f)
end

# Inplace

function InplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return ForwardInplace(f, rec_P, real(x), input_data)
end

function ThreadedInplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return ThreadedInplace(f, rec_P, real(x), input_data)
end

function GPUInplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), input_data = logkernel_init(n, x)

    # enforce Float32 recurrence coefficients on the GPU
    A, B, C = Float32.(A), Float32.(B), Float32.(C)
    return GPUInplace(f, (A, B, C), real(x), input_data)
end


function stieltjes_init(n::Integer, x::AbstractVector, P::OrthogonalPolynomial)
    T = eltype(real(x))

    w = orthogonalityweight(P)
    A, B, C = recurrencecoefficients(P)
    A, B, C = convert.(T, A[1:n]), convert.(T, B[1:n]), convert.(T, C[1:n])

    data = Matrix{T}(undef, 2, length(x))

    data[1, :] .= convert.(T, real(stieltjes(w, x) .* _p0(P)))
    data[2, :] .= convert.(T, real((A[1] .* x .+ B[1]) .* data[1, :] .- (A[1]sum(w) * _p0(P))))
    
    return (A, B, C), data
end

function logkernel_init(n::Integer, x::AbstractVector)
    zlog(z) = ifelse(iszero(z), z, z * log(z))
    T = eltype(real(x))

    A, B, C = recurrencecoefficients(Ultraspherical(-1/2))
    A, B, C = convert.(T, A[2:n]), convert.(T, B[2:n]), convert.(T, C[2:n])

    data = Matrix{T}(undef, 3, length(x))

    data[1, :] .= @. convert(T, real(zlog(1 + x) - zlog(x - 1) - 2one(eltype(x))))
    data[2, :] .= @. convert(T, real((x + 1) * data[1, :]/2 + 1 - zlog(x + 1)))
    data[3, :] .= @. convert(T, real(x * data[2, :] + 2one(eltype(x))/3))
    
    return (A, B, C), data
end