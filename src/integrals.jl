# Forward

function FixedCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = cauchy_init(n, x)
    return transpose(FixedRecurrenceArray(x, rec_P, n, input_data)) * f
end

function ThreadedCauchy(n::Integer, x::AbstractVector, f::AbstractVector, dims::Union{Val{1}, Val{2}})
    rec_P, input_data = cauchy_init(n, x)
    return transpose(ThreadedRecurrenceArray(x, rec_P, n, input_data, dims)) * f
end

function GPUFixedCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = cauchy_init(n, x)
    return transpose(GPURecurrenceArray(x, rec_P, n, input_data)) * CUDA.CuArray(f)
end

function FixedLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(transpose(FixedRecurrenceArray(x, rec_P, n, input_data)) * f)
end

function ThreadedLogKernel(n::Integer, x::AbstractVector, f::AbstractVector, dims::Union{Val{1}, Val{2}})
    rec_P, input_data = logkernel_init(n, x)
    return real.(transpose(ThreadedRecurrenceArray(x, rec_P, n, input_data, dims)) * f)
end

function GPUFixedLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(transpose(GPURecurrenceArray(x, rec_P, n, input_data)) * CUDA.CuArray(f))
end

# Clenshaw

function ClenshawCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, data = cauchy_init(n + 1, x)
    return FixedClenshaw(f, rec_P, x, view(data, 1, :), view(data, 2, :))
end

function ThreadedClenshawCauchy(n::Integer, x::AbstractVector, f::AbstractVector, dims::Union{Val{1}, Val{2}})
    rec_P, data = cauchy_init(n + 1, x)
    return ThreadedClenshaw(f, rec_P, x, view(data, 1, :), view(data, 2, :), dims)
end

function GPUClenshawCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), data = cauchy_init(n + 1, x)
    return GPUClenshaw(f, (Float32.(A), Float32.(B), Float32.(C)), x, view(data, 1, :), view(data, 2, :))
end

function ClenshawLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), data = logkernel_init(n + 2, x)
    return real.(f[1] * view(data, 1, :) .+ FixedClenshaw(f[2:end], (A[2:end], B[2:end], C[2:end]), x, view(data, 2, :), view(data, 3, :)))
end

function ThreadedClenshawLogKernel(n::Integer, x::AbstractVector, f::AbstractVector, dims::Union{Val{1}, Val{2}})
    (A, B, C), data = logkernel_init(n + 2, x)
    return real.(f[1] * view(data, 1, :) .+ ThreadedClenshaw(f[2:end], (A[2:end], B[2:end], C[2:end]), x, view(data, 2, :), view(data, 3, :), dims))
end

function GPUClenshawLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), data = logkernel_init(n + 2, x)
    return real.(CUDA.CuArray(f[1] * view(data, 1, :)) .+ GPUClenshaw(f[2:end], (Float32.(A[2:end]), Float32.(B[2:end]), Float32.(C[2:end])), x, view(data, 2, :), view(data, 3, :)))
end

# Inplace

function InplaceCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = cauchy_init(n, x)
    return ForwardInplace(f, rec_P, x, input_data)
end

function ThreadedInplaceCauchy(n::Integer, x::AbstractVector, f::AbstractVector, dims::Union{Val{1}, Val{2}})
    rec_P, input_data = cauchy_init(n, x)
    return ThreadedInplace(f, rec_P, x, input_data, dims)
end

function GPUInplaceCauchy(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), input_data = cauchy_init(n, x)
    return GPUInplace(f, (Float32.(A), Float32.(B), Float32.(C)), x, input_data)
end

function InplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    rec_P, input_data = logkernel_init(n, x)
    return real.(ForwardInplace(f, rec_P, x, input_data))
end

function ThreadedInplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector, dims::Union{Val{1}, Val{2}})
    rec_P, input_data = logkernel_init(n, x)
    return real.(ThreadedInplace(f, rec_P, x, input_data, dims))
end

function GPUInplaceLogKernel(n::Integer, x::AbstractVector, f::AbstractVector)
    (A, B, C), input_data = logkernel_init(n, x)
    return real.(GPUInplace(f, (Float32.(A), Float32.(B), Float32.(C)), x, input_data))
end

function cauchy_init(n::Integer, x::AbstractVector)
    T = eltype(x)
    P = Legendre()

    w = orthogonalityweight(P)
    A, B, C = recurrencecoefficients(P)
    A, B, C = convert.(T, A[1:n]), convert.(T, B[1:n]), convert.(T, C[1:n])

    data = Matrix{T}(undef, 2, length(x))

    data[1, :] .= -stieltjes(w, x) .* _p0(P)
    data[2, :] .= (A[1] .* x .+ B[1]) .* data[1, :] .+ (A[1]sum(w) * _p0(P))

    return (A, B, C), data
end

function logkernel_init(n::Integer, x::AbstractVector)
    zlog(z) = ifelse(iszero(z), z, z * log(z))
    T = eltype(x)

    A, B, C = recurrencecoefficients(Ultraspherical(-1 / 2))
    A, B, C = convert.(real(T), A[2:n]), convert.(real(T), B[2:n]), convert.(real(T), C[2:n])

    data = Matrix{T}(undef, 3, length(x))

    data[1, :] .= @. zlog(1 + x) - zlog(x - 1) - 2one(T)
    data[2, :] .= @. (x + 1) * data[1, :] / 2 + 1 - zlog(x + 1)
    data[3, :] .= @. x * data[2, :] + 2one(T) / 3

    return (A, B, C), data
end