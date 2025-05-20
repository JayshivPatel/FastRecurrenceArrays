# public

FixedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector) =
    Clenshaw(c, (A, B, C), x, nothing, nothing)

FixedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector) =
    Clenshaw(c, (A, B, C), x, p0, (A[1] .* x .+ B[1]) .* p0)

FixedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector) =
    Clenshaw(c, (A, B, C), x, p0, p1)

ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, dims::Val{1}) =
    Clenshaw(c, (A, B, C), x, nothing, nothing, rowthreadedclenshaw!)

ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, dims::Val{1}) =
    Clenshaw(c, (A, B, C), x, p0, (A[1] .* x .+ B[1]) .* p0, rowthreadedclenshaw!)

ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector, dims::Val{1}) =
    Clenshaw(c, (A, B, C), x, p0, p1, rowthreadedclenshaw!)

ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, dims::Val{2}) =
    Clenshaw(c, (A, B, C), x, nothing, nothing, columnthreadedclenshaw!)

ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, dims::Val{2}) =
    Clenshaw(c, (A, B, C), x, p0, (A[1] .* x .+ B[1]) .* p0, columnthreadedclenshaw!)

ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector, dims::Val{2}) =
    Clenshaw(c, (A, B, C), x, p0, p1, columnthreadedclenshaw!)

GPUClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector) =
    _GPUClenshaw(c, (A, B, C), x, nothing, nothing)

GPUClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector) =
    _GPUClenshaw(c, (A, B, C), x, p0, (A[1] .* x .+ B[1]) .* p0)

GPUClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector) =
    _GPUClenshaw(c, (A, B, C), x, p0, p1)

# protected

function Clenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::Union{AbstractVector,Nothing}, p1::Union{AbstractVector,Nothing}, populate!::Function=clenshaw!)
    f = Vector{eltype(x)}(undef, length(x))

    populate!(f, c, (A, B, C), x, p0, p1)

    return f
end

function _GPUClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::Nothing, p1::Nothing)
    # enforce Float32/ComplexF32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)

    num_points = length(x)
    num_coeffs = length(c)

    T = eltype(x)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return CUDA.zeros(T, num_points)

    gpu_x = CuArray(x)
    gpu_bn2 = CUDA.zeros(T, num_points)
    gpu_bn1 = CUDA.fill(convert(T, c[num_coeffs]), num_points)
    gpu_next = CuArray{T}(undef, num_points)
    gpu_f = CuArray{T}(undef, num_points)

    num_coeffs == 1 && return gpu_bn1

    @inbounds for n = num_coeffs-1:-1:2
        # bₙ(x) = fₙ + (Aₙx + Bₙ)bₙ₊₁(x) - Cₙ₊₁bₙ₊₂(x) 
        @. gpu_next = c[n] + (A[n] * gpu_x + B[n]) * gpu_bn1 - C[n+1] * gpu_bn2

        @. gpu_bn2 = gpu_bn1
        @. gpu_bn1 = gpu_next
    end

    @. gpu_f = c[1] + (A[1] * gpu_x + B[1]) * gpu_bn1 - C[2] * gpu_bn2

    return gpu_f
end

function _GPUClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector)
    # enforce Float32/ComplexF32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)
    p0 = checkandconvert(p0)
    p1 = checkandconvert(p1)

    num_points = length(x)
    num_coeffs = length(c)

    T = eltype(x)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return CUDA.zeros(T, num_points)

    gpu_x = CuArray(x)
    gpu_bn2 = CUDA.zeros(T, num_points)
    gpu_bn1 = CUDA.fill(convert(T, c[num_coeffs]), num_points)
    gpu_next = CuArray{T}(undef, num_points)
    gpu_f = CuArray{T}(undef, num_points)
    gpu_p0 = CuArray(p0)
    gpu_p1 = CuArray(p1)

    num_coeffs == 1 && return gpu_bn1

    @inbounds for n = num_coeffs-1:-1:2
        @. gpu_next = c[n] + (A[n] * gpu_x + B[n]) * gpu_bn1 - C[n+1] * gpu_bn2

        @. gpu_bn2 = gpu_bn1
        @. gpu_bn1 = gpu_next
    end

    @. gpu_next = c[1] + (A[1] * gpu_x + B[1]) * gpu_bn1 - C[2] * gpu_bn2

    # fₓ ≈ b₀(x)p₀(x) + b₁(x)(p₁(x) - α₀(x)p₀(x))
    @. gpu_f = (gpu_next * gpu_p0) + gpu_bn1 * (gpu_p1 - (A[1] * gpu_x + B[1]) * gpu_p0)

    return gpu_f
end

function clenshaw!(f::AbstractVector, c::AbstractVector, (A, B, C), x::AbstractVector, p0::Union{AbstractVector,Nothing}, p1::Union{AbstractVector,Nothing})
    f .= clenshaw_data.(Ref(c), Ref((A, B, C)), x, p0, p1)
end

function rowthreadedclenshaw!(f::AbstractVector, c::AbstractVector, (A, B, C), x::AbstractVector, p0::Nothing, p1::Nothing)
    num_points = length(x)
    num_coeffs = length(c)

    T = eltype(x)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(T)

    bn2 = Base.zeros(T, num_points)
    bn1 = Base.fill(convert(T, c[num_coeffs]), num_points)
    next = Array{T}(undef, num_points)

    num_coeffs == 1 && return bn1

    @inbounds for n = num_coeffs-1:-1:2
        Threads.@threads for j in axes(x, 1)
            next[j] = c[n] + (A[n] * x[j] + B[n]) * bn1[j] - C[n+1] * bn2[j]
        end

        @. bn2 = bn1
        @. bn1 = next
    end

    @. f = c[1] + (A[1] * x + B[1]) * bn1 - C[2] * bn2
end

function rowthreadedclenshaw!(f::AbstractVector, c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector)
    num_points = length(x)
    num_coeffs = length(c)

    T = eltype(x)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(T)

    bn2 = Base.zeros(T, num_points)
    bn1 = Base.fill(convert(T, c[num_coeffs]), num_points)
    next = Array{T}(undef, num_points)

    num_coeffs == 1 && return bn1

    @inbounds for n = num_coeffs-1:-1:2
        Threads.@threads for j in axes(x, 1)
            next[j] = c[n] + (A[n] * x[j] + B[n]) * bn1[j] - C[n+1] * bn2[j]
        end

        @. bn2 = bn1
        @. bn1 = next
    end

    @. next = c[1] + (A[1] * x + B[1]) * bn1 - C[2] * bn2

    @. f = (next * p0) + bn1 * (p1 - (A[1] * x + B[1]) * p0)
end

function columnthreadedclenshaw!(f::AbstractVector, c::AbstractVector, (A, B, C), x::AbstractVector, p0::Nothing, p1::Nothing)
    @inbounds Threads.@threads for j in axes(x, 1)
        f[j] = clenshaw_data(c, (A, B, C), x[j], p0, p1)
    end
end

function columnthreadedclenshaw!(f::AbstractVector, c::AbstractVector, (A, B, C), x::AbstractVector, p0::AbstractVector, p1::AbstractVector)
    @inbounds Threads.@threads for j in axes(x, 1)
        f[j] = clenshaw_data(c, (A, B, C), x[j], p0[j], p1[j])
    end
end

function clenshaw_data(c::AbstractVector, (A, B, C), x::Number, p0::Number, p1::Number)
    num_coeffs = length(c)

    T = eltype(x)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return zero(T)

    @inbounds begin
        bn2 = zero(T)
        bn1 = convert(T, c[num_coeffs])

        num_coeffs == 1 && return bn1

        for n = num_coeffs-1:-1:2
            bn1, bn2 = clenshaw_next(n, A, B, C, x, c, bn1, bn2), bn1
        end

        bn1, bn2 = _clenshaw_first(A, B, C, x, c, bn1, bn2), bn1
    end

    # fₓ ≈ b₀(x)p₀(x) + b₁(x)(p₁(x) - α₀(x)p₀(x))
    return (bn1 * p0) + bn2 * (p1 - (A[1] * x + B[1]) * p0)
end

clenshaw_data(c::AbstractVector, (A, B, C), x::Number, p0::Nothing, p1::Nothing) =
    clenshaw(c, A, B, C, x)