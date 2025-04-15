import Base: axes, getindex, size, show
import RecurrenceRelationships: check_clenshaw_recurrences, clenshaw_next, _clenshaw_first
import CUDA

export FixedClenshaw, ThreadedClenshaw, GPUClenshaw

# struct

struct FixedClenshaw{Coefs<:AbstractVector,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector,XX<:AbstractVector}
    c::Coefs
    A::AA
    B::BB
    C::CC
    x::XX
    f::XX
    p0::XX
    p1::XX
end

# constructors

function FixedClenshaw(c::AbstractVector, A, B, C, x::AbstractVector,
    p0::AbstractVector=ones(eltype(x), length(x)), p1::AbstractVector=(A[1] .* x .+ B[1]) .* p0, computeClenshaw::Function=serialclenshaw)

    num_coeffs = length(c)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(eltype(x))

    # calculate fₓ using Clenshaw's
    fₓ = computeClenshaw(x, c, A, B, C, p0, p1)

    return FixedClenshaw(c, A, B, C, x, fₓ, p0, p1)
end

# dim 1: rows, dim 2: columns

function ThreadedClenshaw(c::AbstractVector, A, B, C, x::AbstractVector, dims::Integer=2,
    p0::AbstractVector=ones(eltype(x), length(x)), p1::AbstractVector=(A[1] .* x .+ B[1]) .* p0)

    @assert dims == 1 || dims == 2 "dimension must be either 1 or 2."

    if dims == 1
        return FixedClenshaw(c, A, B, C, x, p0, p1, rowthreadedclenshaw)
    elseif dims == 2
        return FixedClenshaw(c, A, B, C, x, p0, p1, columnthreadedclenshaw)
    end
end

function GPUClenshaw(c::AbstractVector, A, B, C, x::AbstractVector,
    p0::AbstractVector=ones(eltype(x), length(x)), p1::AbstractVector=(A[1] .* x .+ B[1]) .* p0)

    # enforce Float32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)
    p0 = checkandconvert(p0)
    p1 = checkandconvert(p1)

    fₓ = gpuclenshaw(x, c, A, B, C, p0, p1)

    return FixedClenshaw(c, A, B, C, x, fₓ, p0, p1)
end

# properties and access

copy(M::FixedClenshaw) = M # immutable entries
size(M::FixedClenshaw) = size(M.f)
axes(M::FixedClenshaw) = axes(M.f)
getindex(M::FixedClenshaw, index...) = M.f[index...]

# display

function show(io::IO, ::MIME"text/plain", M::FixedClenshaw)
    s = size(M)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(M)) * ":"
    )
    show(io, MIME"text/plain"(), M.f)
end

# serial clenshaw

function serialclenshaw(x::AbstractVector, c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)
    num_points = length(x)
    num_coeffs = length(c)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(eltype(x))

    b₀ = Base.zeros(eltype(x), length(x))
    b₁ = Base.zeros(eltype(x), length(x))

    num_coeffs == 1 && return bn1

    @inbounds for j in axes(x, 1)
        bn2 = zero(eltype(x))
        bn1 = convert(eltype(x), c[num_coeffs])
        for n = num_coeffs-1:-1:2
            bn1, bn2 = clenshaw_next(n, A, B, C, x[j], c, bn1, bn2), bn1
        end

        b₀[j] = _clenshaw_first(A, B, C, x[j], c, bn1, bn2)
        b₁[j] = bn1
    end

    # fₓ ≈ b₀(x)p₀(x) + b₁(x)(p₁(x) - α₀(x)p₀(x))
    return (b₀ .* p0) .+ b₁ .* (p1 .- (A[1] .* x .+ B[1]) .* p0)
end

# column-wise threaded clenshaw

function columnthreadedclenshaw(x::AbstractVector, c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)

    fₓ = Base.zeros(eltype(x), length(x))

    @inbounds Threads.@threads for j in axes(x, 1)
        fₓ[j] = serialclenshaw([x[j]], c, A, B, C, [p0[j]], [p1[j]])[1]
    end

    return fₓ
end

# row-wise threaded clenshaw

function rowthreadedclenshaw(x::AbstractVector, c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)
    num_points = length(x)
    num_coeffs = length(c)

    @inbounds begin
        bn2 = Base.zeros(eltype(x), length(x))
        bn1 = Base.fill(convert(eltype(x), c[num_coeffs]), length(x))
        bn0 = Base.zeros(eltype(x), length(x))

        if num_coeffs == 1
            return bn1
        end

        for i in num_coeffs-1:-1:2
            Threads.@threads for j in axes(x, 1)
                bn1[j], bn2[j] = clenshaw_next(i, A, B, C, x[j], c, bn1[j], bn2[j]), bn1[j]
            end
        end

        Threads.@threads for j in axes(x, 1)
            bn0[j] = _clenshaw_first(A, B, C, x[j], c, bn1[j], bn2[j])
        end

        # fₓ ≈ b₀(x)p₀(x) + b₁(x)(p₁(x) - α₀(x)p₀(x))
        return (bn0 .* p0) .+ bn1 .* (p1 .- (A[1] .* x .+ B[1]) .* p0)
    end
end

# row-wise GPU clenshaw

function gpuclenshaw(x::AbstractVector, c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)

    num_points = length(x)
    num_coeffs = length(c)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(eltype(x))

    @inbounds begin
        # copy the data to the GPU
        gpu_x = CUDA.CuArray(x)
        gpu_p0 = CUDA.CuArray(p0)
        gpu_p1 = CUDA.CuArray(p1)

        # initialise arrays for the clenshaw computation
        gpu_bn2, gpu_bn1 =
            CUDA.zeros(eltype(x), num_points), CUDA.fill(convert(eltype(x), c[num_coeffs]), num_points)

        num_coeffs == 1 && return Array(gpu_bn1)

        for n = num_coeffs-1:-1:2
            gpu_bn1, gpu_bn2 =
                gpuclenshaw_next(A[n], B[n], C[n+1], gpu_x, c[n], gpu_bn1, gpu_bn2, num_points), gpu_bn1
        end

        gpu_bn0 = gpuclenshaw_next(A[1], B[1], C[2], gpu_x, c[1], gpu_bn1, gpu_bn2, num_points)

        A₁ = CUDA.fill(A[1], num_points)
        B₁ = CUDA.fill(B[1], num_points)

        # fₓ ≈ b₀(x)p₀(x) + b₁(x)(p₁(x) - α₀(x)p₀(x))
        fₓ = (gpu_bn0 .* gpu_p0) .+ gpu_bn1 .* (gpu_p1 .- (A₁ .* gpu_x + B₁) .* gpu_p0)

        return Array(fₓ)
    end
end

function gpuclenshaw_next(A, B, C, X::CUDA.CuArray, c,
    bn1::CUDA.CuArray, bn2::CUDA.CuArray, num_points::Integer)

    # construct vectors
    Aₙ = CUDA.fill(A, num_points)
    Bₙ = CUDA.fill(B, num_points)
    Cₙ = CUDA.fill(C, num_points)
    cₙ = CUDA.fill(c, num_points)

    # bₙ(x) = fₙ + (Aₙx + Bₙ)bₙ₊₁(x) - Cₙ₊₁bₙ₊₂(x) 
    return (cₙ + (Aₙ .* X + Bₙ) .* bn1 - Cₙ .* bn2)
end