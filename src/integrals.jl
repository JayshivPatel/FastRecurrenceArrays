import ClassicalOrthogonalPolynomials: Legendre, OrthogonalPolynomial, Ultraspherical,
    orthogonalityweight, recurrencecoefficients, _p0
import LinearAlgebra: dot
import SingularIntegrals: stieltjes

export FixedStieltjes, InplaceStieltjes, ThreadedInplaceStieltjes, GPUInplaceStieltjes,
    FixedLogKernel, InplaceLogKernel, ThreadedInplaceLogKernel, GPUInplaceLogKernel

# Forward

function FixedStieltjes(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = stieltjes_init(n, x)
    return dot(f, FixedRecurrenceArray(x, rec_P, n - 2, input_data))
end

# Inplace

function InplaceStieltjes(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = stieltjes_init(n, x)
    return ForwardInplace(f, rec_P, x, input_data)
end

function ThreadedInplaceStieltjes(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = stieltjes_init(n, x)
    return ThreadedInplace(f, rec_P, x, input_data)
end

function GPUInplaceStieltjes(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), input_data = stieltjes_init(n, x)
    
    # enforce Float32 recurrence coefficients on the GPU
    A, B, C = Float32.(A), Float32.(B), Float32.(C)
    return GPUInplace(f, (A, B, C), x, input_data)
end

# Forward 

function FixedLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(dot(f, FixedRecurrenceArray(x, rec_P, n - 2, input_data)))
end

# Inplace

function InplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(ForwardInplace(f, rec_P, x, input_data))
end

function ThreadedInplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(ThreadedInplace(f, rec_P, x, input_data))
end

function GPUInplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), input_data = logkernel_init(n, x)

    # enforce Float32 recurrence coefficients on the GPU
    A, B, C = Float32.(A), Float32.(B), Float32.(C)
    return real.(GPUInplace(f, (A, B, C), x, input_data))
end

function stieltjes_init(n::Integer, x::AbstractVector)
    T = eltype(x)
    P = Legendre()

    w = orthogonalityweight(P)
    A, B, C = recurrencecoefficients(P)
    A, B, C = convert.(T, A[1:n]), convert.(T, B[1:n]), convert.(T, C[1:n])

    data = Matrix{T}(undef, 2, length(x))

    data[1, :] .= stieltjes(w, x) .* _p0(P)
    data[2, :] .= (A[1] .* x .+ B[1]) .* data[1, :] .- (A[1]sum(w) * _p0(P))
    
    return (A, B, C), data
end

function logkernel_init(n::Integer, x::AbstractVector)
    zlog(z) = ifelse(iszero(z), z, z * log(z))
    T = eltype(x)

    A, B, C = recurrencecoefficients(Ultraspherical(-1/2))
    A, B, C = convert.(real(T), A[2:n]), convert.(real(T), B[2:n]), convert.(real(T), C[2:n])

    data = Matrix{T}(undef, 3, length(x))

    data[1, :] .= @. zlog(1 + x) - zlog(x - 1) - 2one(T)
    data[2, :] .= @. (x + 1) * data[1, :]/2 + 1 - zlog(x + 1)
    data[3, :] .= @. x * data[2, :] + 2one(T)/3
    
    return (A, B, C), data
end