import RecurrenceRelationships: check_clenshaw_recurrences, clenshaw, clenshaw!, clenshaw_next, _clenshaw_first
import BandedMatrices: AbstractBandedMatrix, bandwidth
import Base: axes, getindex, size, show

using CUDA

export FixedClenshaw, ThreadedClenshaw, GPUClenshaw

# struct

struct FixedClenshaw{T,Coefs<:AbstractVector,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector,XX<:AbstractVector}
    c::Coefs
    A::AA
    B::BB
    C::CC
    x::XX
    data::Array{T}
    p0::T
end


# constructors

function FixedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector,
    populate::Function=clenshaw!)

    num_coeffs = length(c)

    T = promote_type(eltype(c), eltype(x))

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(T)

    # copy the initialisation to a struct
    M = FixedClenshaw(convert(AbstractVector{T}, c), A, B, C, convert(AbstractVector{T}, x), Base.copy(x), one(T))

    # calculate and populate the data using Clenshaw's
    populate(M.data, M.c, M.A, M.B, M.C)

    return M
end

FixedClenshaw(c::Number, (A, B, C), X, p) = FixedClenshaw([c], (A, B, C), X, p)
FixedClenshaw(c, (A, B, C), x::Number, p) = FixedClenshaw(c, (A, B, C), [x], p)


# dim 1: rows, dim 2: columns

function ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector,
    dims::Integer=1)

    @assert dims == 1 || dims == 2 "dimension must be either 1 or 2."

    if dims == 1
        return FixedClenshaw(c, (A, B, C), x, rowthreadedclenshaw!)
    elseif dims == 2
        return FixedClenshaw(c, (A, B, C), x, columnthreadedclenshaw!)
    end
end

function GPUClenshaw(c::AbstractVector, (A, B, C), X::AbstractVector)

    if (eltype(c) == Float64 || eltype(A) == Float64 || eltype(B) == Float64 ||
        eltype(C) == Float64 || eltype(X) == Float64)
        @warn "Converting input vector(s) to Float32 for improved performance..."
    end

    # enforce Float32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    X = checkandconvert(X)

    # copy the initialisation to a struct
    M = FixedClenshaw(c, A, B, C, X, Base.copy(X), one(Float32))

    # calculate and populate the data using Clenshaw's on a GPU
    gpuclenshaw!(M)

    return M
end

GPUClenshaw(c::Number, (A, B, C), X, p) = GPUClenshaw([c], (A, B, C), X, p)

# properties and access

copy(M::FixedClenshaw) = M # immutable entries
size(M::FixedClenshaw) = size(M.data)
axes(M::FixedClenshaw) = axes(M.data)
bandwidths(M::FixedClenshaw) = (length(M.c) - 1, length(M.c) - 1)
getindex(M::FixedClenshaw, index...) = M.data[index...]

# display

function show(io::IO, ::MIME"text/plain", M::FixedClenshaw)
    s = size(M)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(M)) * ":"
    )
    show(io, MIME"text/plain"(), M.data)
end

# column-wise threaded clenshaw

function columnthreadedclenshaw!(x::AbstractVector, c::AbstractVector, A, B, C)

    @inbounds Threads.@threads for j in axes(x, 1)
        x[j] = clenshaw(c, A, B, C, x[j])
    end
end

# row-wise threaded clenshaw

function rowthreadedclenshaw!(x::AbstractVector, c::AbstractVector, A, B, C)

    num_points = length(x)
    num_coeffs = length(c)

    T = eltype(x)

    @inbounds begin
        bn2, bn1 = zeros(T, length(x)), fill(convert(T, c[num_coeffs]), length(x))

        if num_coeffs == 1
            x = bn1
            return
        end

        for i in num_coeffs-1:-1:2
            Threads.@threads for j in axes(x, 1)
                bn1[j], bn2[j] = clenshaw_next(i, A, B, C, x[j], c, bn1[j], bn2[j]), bn1[j]
            end
        end

        Threads.@threads for j in axes(x, 1)
            bn1[j] = _clenshaw_first(A, B, C, x[j], c, bn1[j], bn2[j])
        end

        x .= bn1
    end
end

# row-wise GPU clenshaw

function gpuclenshaw!(M::FixedClenshaw)

    num_points = length(M.x)
    num_coeffs = length(M.c)

    T = typeof(M.p0)

    @boundscheck check_clenshaw_recurrences(num_coeffs, M.A, M.B, M.C)

    num_coeffs == 0 && return zero(T)

    @inbounds begin
        # copy the data to the GPU
        gpu_data = CuArray(M.data)

        # initialise arrays for the clenshaw computation
        gpu_bn2, gpu_bn1 = CUDA.zeros(T, num_points), CUDA.fill(convert(T, M.c[num_coeffs]), num_points)
        num_coeffs == 1 && return Array(gpu_bn1)

        for n = num_coeffs-1:-1:2
            gpu_bn1, gpu_bn2 =
                gpuclenshaw_next(n, M.A, M.B, M.C, gpu_data, M.c, gpu_bn1, gpu_bn2, num_points), gpu_bn1
        end

        gpu_bn1 = gpuclenshaw_next(1, M.A, M.B, M.C, gpu_data, M.c, gpu_bn1, gpu_bn2, num_points)
    end

    copyto!(M.data, gpu_bn1)
end

function gpuclenshaw_next(n::Integer, A, B, C, X::CuArray{Float32}, c,
    bn1::CuArray{Float32}, bn2::CuArray{Float32}, num_points::Integer)

    # construct vectors
    Aₙ = CUDA.fill(A[n], num_points)
    Bₙ = CUDA.fill(B[n], num_points)
    Cₙ = CUDA.fill(C[n+1], num_points)
    cₙ = CUDA.fill(c[n], num_points)

    # calculate and return the next recurrence
    return ((Aₙ .* X + Bₙ) .* bn1 - Cₙ .* bn2 + cₙ)
end