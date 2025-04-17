import Base: axes, getindex, size, show
import CUDA

export ForwardInplace,
    ThreadedInplace,
    GPUInplace

# struct

struct ForwardInplace{T,Coefs<:AbstractVector,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector,XX<:AbstractVector, FF<:AbstractVector}
    c::Coefs
    A::AA
    B::BB
    C::CC
    x::XX
    f::FF
    p0::T
end

# constructors

function ForwardInplace(c::AbstractVector, (A, B, C), x::AbstractVector,
    input_data::AbstractMatrix=Base.zeros(eltype(x), 1, length(x)), populate::Function=forwardvec_inplace!)

    num_coeffs = length(c)
    num_points = length(x)

    T = promote_type(eltype(c), eltype(x))

    num_coeffs == 0 && return zero(T)

    M, N = size(input_data)
    fₓ = Base.zeros(T, num_points)

    p0 = Vector{T}(undef, N)
    p1 = Vector{T}(undef, N)

    if M < 2
        for j = axes(x, 1)
            p0[j] = convert(T, one(x[j]))
            p1[j] = convert(T, muladd(A[1], x[j], B[1]) * p0[j])
        end
        fₓ += p0 * c[1] + p1 * c[2]
        M = 2
    else
        for i in 1:M
            fₓ += input_data[i, :] * c[i]
        end
        p0 = input_data[end-1, :]
        p1 = input_data[end, :]
    end

    # calculate and populate fₓ using forward_inplace
    populate(M, fₓ, x, c, A, B, C, p0, p1)

    return ForwardInplace(c, A, B, C, x, fₓ, one(T))
end

function ThreadedInplace(c::AbstractVector, (A, B, C), x::AbstractVector,
    input_data::AbstractMatrix=Base.zeros(eltype(x), 1, length(x)))

    return ForwardInplace(c, (A, B, C), x, input_data, threaded_inplace!)
end

function GPUInplace(c::AbstractVector, (A, B, C), x::AbstractVector,
    input_data::AbstractMatrix=Base.zeros(Float32, 1, length(x)))

    # enforce Float32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)
    input_data = checkandconvert(input_data)

    return ForwardInplace(c, (A, B, C), x, input_data, gpu_inplace!)
end

# display

function show(io::IO, ::MIME"text/plain", M::ForwardInplace)
    s = size(M)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(M)) * ":"
    )
    show(io, MIME"text/plain"(), M.f)
end

# properties and access

copy(M::ForwardInplace) = M # immutable entries
size(M::ForwardInplace) = size(M.f)
axes(M::ForwardInplace) = axes(M.f)
getindex(M::ForwardInplace, index...) = M.f[index...]

# serial population

function forwardvec_inplace!(start_index::Integer, f::AbstractVector, x::AbstractVector, 
    c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)

    @inbounds for j in axes(x, 1)
        f[j] += forward_inplace(start_index, c, A, B, C, x[j], p0[j], p1[j])
    end
end

# threaded population (column)

function threaded_inplace!(start_index::Integer, f::AbstractVector, x::AbstractVector, 
    c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)

    @inbounds Threads.@threads for j in axes(x, 1)
        f[j] += forward_inplace(start_index, c, A, B, C, x[j], p0[j], p1[j])
    end
end

# gpu population 
function gpu_inplace!(start_index::Integer, f::AbstractVector, x::AbstractVector, 
    c::AbstractVector, A, B, C, p0::AbstractVector, p1::AbstractVector)

    num_coeffs = length(c)
    num_points = length(x)

    @inbounds begin
        # copy the data to the GPU
        gpu_x = CUDA.CuArray(x)
        gpu_fₓ = CUDA.CuArray(f)

        # initialise arrays for the clenshaw computation
        gpu_p0, gpu_p1 = CUDA.CuArray(p0), CUDA.CuArray(p1)

        num_coeffs == 1 && return Array(gpu_bn1)

        for n = start_index:num_coeffs-1
            gpu_p1, gpu_p0 = gpuforwardrecurrence_next(A[n], B[n], C[n], gpu_x, gpu_p0, gpu_p1, num_points), gpu_p1
            gpu_fₓ += (gpu_p1 .* c[n+1])
        end
    end

    copyto!(f, gpu_fₓ)
end

function forward_inplace(start_index::Integer, c::AbstractVector, A, B, C, 
    x::Number, p0::Number, p1::Number)
    num_coeffs = length(c)

    fₓ = zero(eltype(x))

    @inbounds for i in start_index:num_coeffs-1
        p1, p0 = muladd(muladd(A[i], x, B[i]), p1, -C[i] * p0), p1
        fₓ += p1 * c[i+1]
    end

    return fₓ
end