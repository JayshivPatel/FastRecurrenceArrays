import LinearAlgebra: dot

export ForwardInplace, ThreadedInplace, GPUInplace

# constructors

ForwardInplace(c::AbstractVector, (A, B, C), x::AbstractVector) =
    _ForwardInplace(c, (A, B, C), x, Base.zeros(eltype(x), 1, length(x)))

ForwardInplace(c::AbstractVector, (A, B, C), x::AbstractVector, input_data::AbstractMatrix{T}) where T =
    _ForwardInplace(c, (A, B, C), x, input_data)

ThreadedInplace(c::AbstractVector, (A, B, C), x::AbstractVector, dims::Val{1}) =
    _ForwardInplace(c, (A, B, C), x, Base.zeros(eltype(x), 1, length(x)), rowthreadedinplace!)

ThreadedInplace(c::AbstractVector, (A, B, C), x::AbstractVector, input_data::AbstractMatrix{T}, dims::Val{1}) where T =
    _ForwardInplace(c, (A, B, C), x, input_data, rowthreadedinplace!)

ThreadedInplace(c::AbstractVector, (A, B, C), x::AbstractVector, dims::Val{2}) =
    _ForwardInplace(c, (A, B, C), x, Base.zeros(eltype(x), 1, length(x)), columnthreadedinplace!)

ThreadedInplace(c::AbstractVector, (A, B, C), x::AbstractVector, input_data::AbstractMatrix{T}, dims::Val{2}) where T =
    _ForwardInplace(c, (A, B, C), x, input_data, columnthreadedinplace!)

GPUInplace(c::AbstractVector, (A, B, C), x::AbstractVector) =
    _GPUInplace(c, (A, B, C), x, Base.zeros(eltype(x), 1, length(x)))

GPUInplace(c::AbstractVector, (A, B, C), x::AbstractVector, input_data::AbstractMatrix{T}) where T =
    _GPUInplace(c, (A, B, C), x, input_data)


function _ForwardInplace(c::AbstractVector, (A, B, C), x::AbstractVector,
    input_data::AbstractMatrix{T}, populate!::Function=forwardvec_inplace!) where T

    num_coeffs = length(c)
    num_points = length(x)

    @assert num_coeffs >= 2

    N, _ = size(input_data)

    p0 = Vector{T}(undef, num_points)
    p1 = Vector{T}(undef, num_points)
    f = Vector{T}(undef, num_points)

    if N < 2
        @. p0 = Base.one(T)
        @. p1 = (A[1] * x + B[1]) * Base.one(T)
        @. f = p0 * c[1] + p1 * c[2]
        N = 2
    else
        p0 = input_data[end-1, :]
        p1 = input_data[end, :]

        f = sum(view(input_data, i, :) * c[i] for i in 1:N)
    end

    # calculate and populate f using forward_inplace
    populate!(N, f, x, c, (A, B, C), p0, p1)

    return f
end

function _GPUInplace(c::AbstractVector, (A, B, C), x::AbstractVector, input_data::AbstractMatrix{T}) where T

    # enforce Float32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    x = checkandconvert(x)
    input_data = checkandconvert(input_data)

    num_coeffs = length(c)
    num_points = length(x)

    @assert num_coeffs >= 2

    N, _ = size(input_data)
    gpu_f = CuArray{T}(undef, num_points)

    # copy the data to the GPU
    gpu_x = CuArray(x)
    gpu_input_data = CuArray(input_data)

    gpu_p0 = CuArray{eltype(x)}(undef, num_points)
    gpu_p1 = CuArray{eltype(x)}(undef, num_points)
    gpu_next = CuArray{eltype(x)}(undef, num_points)

    if N < 2
        @. gpu_p0 = CUDA.one(T)
        @. gpu_p1 = (A[1] * gpu_x + B[1]) * CUDA.one(T)

        @. gpu_f = c[1] * gpu_p0 + c[1] * gpu_p1
        N = 2
    else
        @. gpu_p0 = gpu_input_data[end-1, :]
        @. gpu_p1 = gpu_input_data[end, :]

        gpu_f = sum(view(gpu_input_data, i, :) .* c[i] for i in 1:N)
    end

    @inbounds for n = N:num_coeffs-1
        @. gpu_next = (A[n] * gpu_x + B[n]) * gpu_p1 - C[n] * gpu_p0
        
        @. gpu_p0 = gpu_p1
        @. gpu_p1 = gpu_next

        @. gpu_f += (c[n+1] * gpu_p1)
    end

    return gpu_f
end

function forwardvec_inplace!(start_index::Integer, f::AbstractVector, x::AbstractVector, c::AbstractVector, (A, B, C), p0::AbstractVector, p1::AbstractVector)
    @inbounds for j in axes(x, 1)
        f[j] += forward_inplace(start_index, c, (A, B, C), x[j], p0[j], p1[j])
    end
end

function rowthreadedinplace!(start_index::Integer, f::AbstractVector, x::AbstractVector, c::AbstractVector, (A, B, C), p0::AbstractVector, p1::AbstractVector)
    num_coeffs = length(c)
    num_points = length(x)
    
    next = Vector{eltype(x)}(undef, num_points)
    
    @inbounds for n = start_index:num_coeffs-1
        Threads.@threads for j in axes(x, 1)
            next[j] = (A[n] * x[j] + B[n]) * p1[j] - C[n] * p0[j]
        end

        @. p0 = p1
        @. p1 = next

        @. f += (c[n+1] * p1)
    end
end

function columnthreadedinplace!(start_index::Integer, f::AbstractVector, x::AbstractVector, c::AbstractVector, (A, B, C), p0::AbstractVector, p1::AbstractVector)
    @inbounds Threads.@threads for j in axes(x, 1)
        f[j] += forward_inplace(start_index, c, (A, B, C), x[j], p0[j], p1[j])
    end
end

function forward_inplace(start_index::Integer, c::AbstractVector, (A, B, C), x::Number, p0, p1)
    num_coeffs = length(c)
    f = zero(typeof(x))

    @inbounds for i in start_index:num_coeffs-1
        p1, p0 = muladd(muladd(A[i], x, B[i]), p1, -C[i] * p0), p1
        f += p1 * c[i+1]
    end

    return f
end