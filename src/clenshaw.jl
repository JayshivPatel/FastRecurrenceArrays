import RecurrenceRelationships: check_clenshaw_recurrences, clenshaw_next, _clenshaw_first, clenshaw
import CUDA: CuArray

export FixedClenshaw, ThreadedClenshaw, GPUClenshaw

# constructors

FixedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector) = clenshaw(c, A, B, C, x)

function ThreadedClenshaw(c::AbstractVector, (A, B, C), x::AbstractVector)
    f = Vector{eltype(x)}(undef, length(x))

    @inbounds Threads.@threads for j in axes(x, 1)
        f[j] = clenshaw(c, A, B, C, x[j])
    end
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
    gpu_x, gpu_A, gpu_B, gpu_C, gpu_c = CuArray(x), CuArray(A), CuArray(B), CuArray(C), CuArray(c)

    # initialise arrays for the clenshaw computation
    gpu_bn2 = CUDA.zeros(eltype(x), num_points)
    gpu_bn1 = CUDA.fill(convert(eltype(x), c[num_coeffs]), num_points)
    gpu_next = CuArray{eltype(x)}(undef, num_points)
    gpu_fₓ = CuArray{eltype(x)}(undef, num_points)

    num_coeffs == 1 && return Array(gpu_bn1)

    @inbounds for n = num_coeffs-1:-1:2
        # bₙ(x) = fₙ + (Aₙx + Bₙ)bₙ₊₁(x) - Cₙ₊₁bₙ₊₂(x) 
        gpu_next .= view(gpu_c, n) .+ (view(gpu_A, n) .* gpu_x .+ view(gpu_B, n)) .* gpu_bn1 .- view(gpu_C, n+1) .* gpu_bn2

        @. gpu_bn2 = gpu_bn1
        @. gpu_bn1 = gpu_next
    end

    gpu_fₓ .= view(gpu_c, 1) .+ (view(gpu_A, 1) .* gpu_x .+ view(gpu_B, 1)) .* gpu_bn1 .- view(gpu_C, 2) .* gpu_bn2

    return gpu_fₓ
end