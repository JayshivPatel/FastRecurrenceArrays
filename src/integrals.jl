import ClassicalOrthogonalPolynomials: Legendre, OrthogonalPolynomial, Ultraspherical,
    orthogonalityweight, recurrencecoefficients, _p0
import LinearAlgebra: dot
import SingularIntegrals: stieltjes

export FixedCauchy, InplaceCauchy, ThreadedInplaceCauchy, GPUInplaceCauchy,
    FixedLogKernel, InplaceLogKernel, ThreadedInplaceLogKernel, GPUInplaceLogKernel

# Forward

function FixedCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = cauchy_init(n, x)
    return transpose(FixedRecurrenceArray(x, rec_P, n, input_data)) * f
end

# Inplace

function InplaceCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = cauchy_init(n, x)
    return ForwardInplace(f, rec_P, x, input_data)
end

function ThreadedInplaceCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = cauchy_init(n, x)
    return ThreadedInplace(f, rec_P, x, input_data)
end

function GPUInplaceCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), input_data = cauchy_init(n, x)
    
    # enforce Float32 recurrence coefficients on the GPU
    A, B, C = Float32.(A), Float32.(B), Float32.(C)
    return GPUInplace(f, (A, B, C), x, input_data)
end

# Forward 

function FixedLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(transpose(FixedRecurrenceArray(x, rec_P, n, input_data)) * f)
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

function cauchy_init(n::Integer, x::AbstractVector)
    T = eltype(x)
    P = Legendre()

    w = orthogonalityweight(P)
    A, B, C = recurrencecoefficients(P)
    A, B, C = convert.(T, A[1:n]), convert.(T, B[1:n]), convert.(T, C[1:n])

    data = Matrix{T}(undef, 2, length(x))

    data[1, :] .= inv(2π*im) * -stieltjes(w, x) .* _p0(P)
    data[2, :] .= (A[1] .* x .+ B[1]) .* data[1, :] .+ (inv(2π*im) * A[1]sum(w) * _p0(P))
    
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