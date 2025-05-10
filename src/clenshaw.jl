import Base: axes, getindex, size, show
import RecurrenceRelationships: check_clenshaw_recurrences, clenshaw_next, _clenshaw_first
import CUDA: CuArray, fill, zeros

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
    p0::AbstractVector=ones(eltype(x), length(x)), p1::AbstractVector=(@.((A[1] * x + B[1]) * p0)), computeClenshaw::Function=serialclenshaw!)

    num_coeffs = length(c)
    num_points = length(x)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(eltype(x))

    fₓ = Base.zeros(eltype(x), num_points)

    # calculate fₓ using Clenshaw's
    computeClenshaw(fₓ, x, c, A, B, C, p0, p1)

    return FixedClenshaw(c, A, B, C, x, fₓ, p0, p1)
end

function ThreadedClenshaw(c::AbstractVector, A, B, C, x::AbstractVector,
    p0::AbstractVector=ones(eltype(x), length(x)), p1::AbstractVector=@.((A[1] * x + B[1]) * p0))

    return FixedClenshaw(c, A, B, C, x, p0, p1, threadedclenshaw)
end

function GPUClenshaw(c::AbstractVector, A, B, C, x::AbstractVector,
    p0::AbstractVector=ones(eltype(x), length(x)), p1::AbstractVector=@.((A[1] * x + B[1]) * p0))

    # enforce Float32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)
    p0 = checkandconvert(p0)
    p1 = checkandconvert(p1)

    fₓ = Base.zeros(eltype(x), length(x))

    gpuclenshaw!(fₓ, x, c, A, B, C, p0, p1)

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

function serialclenshaw!(f::AbstractVector, x::AbstractVector, c::AbstractVector, 
    A, B, C, p0::AbstractVector, p1::AbstractVector)

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
    @. f = ((b₀ * p0) + b₁ * (p1 - (A[1] * x + B[1]) * p0))
end

# threaded

function threadedclenshaw!(f::AbstractVector, x::AbstractVector, c::AbstractVector, 
    A, B, C, p0::AbstractVector, p1::AbstractVector)

    @inbounds Threads.@threads for j in axes(x, 1)
        serialclenshaw!([f[j]], [x[j]], c, A, B, C, [p0[j]], [p1[j]])
    end
end


# GPU

function gpuclenshaw!(f::AbstractVector, x::AbstractVector, c::AbstractVector, 
    A, B, C, p0::AbstractVector, p1::AbstractVector)

    num_points = length(x)
    num_coeffs = length(c)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(eltype(x))

    @inbounds begin
        # copy the data to the GPU
        gpu_x = CuArray(x)
        gpu_p0 = CuArray(p0)
        gpu_p1 = CuArray(p1)

        # initialise arrays for the clenshaw computation
        gpu_bn2 = zeros(eltype(x), num_points)
        gpu_bn1 = fill(convert(eltype(x), c[num_coeffs]), num_points)
        gpu_next = CuArray{eltype(x)}(undef, num_points)
        
        gpu_fₓ = CuArray{eltype(x)}(undef, num_points)

        num_coeffs == 1 && return Array(gpu_bn1)

        for n = num_coeffs-1:-1:2
            gpuclenshaw_next!(gpu_next, A[n], B[n], C[n+1], gpu_x, c[n], gpu_bn1, gpu_bn2)
            
            @. gpu_bn2 = gpu_bn1
            @. gpu_bn1 = gpu_next
        end

        gpuclenshaw_next!(gpu_next, A[1], B[1], C[2], gpu_x, c[1], gpu_bn1, gpu_bn2)

        # fₓ ≈ b₀(x)p₀(x) + b₁(x)(p₁(x) - α₀(x)p₀(x))
        @. gpu_fₓ = (gpu_next * gpu_p0) + gpu_bn1 * (gpu_p1 - (A[1] * gpu_x + B[1]) * gpu_p0)

        copyto!(f, gpu_fₓ)
    end
end

function gpuclenshaw_next!(output::CuArray, A, B, C, X::CuArray, c,
    bn1::CuArray, bn2::CuArray)

    # bₙ(x) = fₙ + (Aₙx + Bₙ)bₙ₊₁(x) - Cₙ₊₁bₙ₊₂(x) 
    @. output = c + (A * X + B) * bn1 - C * bn2
end