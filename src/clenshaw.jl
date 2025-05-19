import RecurrenceRelationships: check_clenshaw_recurrences, clenshaw_next, _clenshaw_first, clenshaw
import CUDA: CuArray

export FixedClenshaw, ThreadedClenshaw, GPUClenshaw

# constructors

FixedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector) = clenshaw(c, A, B, C, x)

function ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, dims::Val{1})
    f = Vector{eltype(x)}(undef, length(x))

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
        # bₙ(x) = fₙ + (Aₙx + Bₙ)bₙ₊₁(x) - Cₙ₊₁bₙ₊₂(x) 
        Threads.@threads for j in axes(x, 1)
            next[j] = c[n] + (A[n] * x[j] + B[n]) * bn1[j] - C[n+1] * bn2[j]
        end
        
        @. bn2 = bn1
        @. bn1 = next
    end

    @. f = c[1] + (A[1] * x + B[1]) * bn1 - C[2] * bn2
end

function ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector, dims::Val{2})
    f = Vector{eltype(x)}(undef, length(x))

    @inbounds Threads.@threads for j in axes(x, 1)
        f[j] = clenshaw(c, A, B, C, x[j])
    end

    return f
end

function GPUClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector)

    # enforce Float32/ComplexF32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)

    num_points = length(x)
    num_coeffs = length(c)

    @boundscheck check_clenshaw_recurrences(num_coeffs, A, B, C)

    num_coeffs == 0 && return CUDA.zeros(eltype(x), num_points)

    # copy the data to the GPU
    gpu_x = CuArray(x)

    # initialise arrays for the clenshaw computation
    gpu_bn2 = CUDA.zeros(eltype(x), num_points)
    gpu_bn1 = CUDA.fill(convert(eltype(x), c[num_coeffs]), num_points)
    gpu_next = CuArray{eltype(x)}(undef, num_points)
    gpu_fₓ = CuArray{eltype(x)}(undef, num_points)

    num_coeffs == 1 && return Array(gpu_bn1)

    @inbounds for n = num_coeffs-1:-1:2
        # bₙ(x) = fₙ + (Aₙx + Bₙ)bₙ₊₁(x) - Cₙ₊₁bₙ₊₂(x) 
        @. gpu_next = c[n] + (A[n] * gpu_x + B[n]) * gpu_bn1 - C[n+1] * gpu_bn2

        @. gpu_bn2 = gpu_bn1
        @. gpu_bn1 = gpu_next
    end

    @. gpu_fₓ = c[1] + (A[1] * gpu_x + B[1]) * gpu_bn1 - C[2] * gpu_bn2

    return gpu_fₓ
end