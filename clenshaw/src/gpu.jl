import RecurrenceRelationships: clenshaw_next, _clenshaw_first, check_clenshaw_recurrences

using CUDA

export GPUFixedClenshaw

# constructor

function GPUFixedClenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector,
    C::AbstractVector, X::AbstractVector)

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
    M = FixedClenshaw(c, A, B, C, X, copy(X), one(Float32))

    # calculate and populate the data using Clenshaw's on a GPU
    gpuclenshaw!(M)
end

GPUFixedClenshaw(c::Number, A, B, C, X, p) = GPUFixedClenshaw([c], A, B, C, X, p)
GPUFixedClenshaw(c, A, B, C, x::Number, p) = GPUFixedClenshaw(c, A, B, C, [x], p)

function gpuclenshaw!(M::FixedClenshaw)
    num_points = length(M)
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
                clenshaw_next(n, M.A, M.B, M.C, gpu_data, M.c, gpu_bn1, gpu_bn2, num_points), gpu_bn1
        end

        gpu_bn1 = clenshaw_next(1, M.A, M.B, M.C, gpu_data, M.c, gpu_bn1, gpu_bn2, num_points)
    end

    return gpu_bn1
end

# clenshaw

function clenshaw_next(n::Integer, A, B, C, X::CuArray{Float32}, c, 
    bn1::CuArray{Float32}, bn2::CuArray{Float32}, num_points::Integer)
    # construct vectors
    Aₙ = CUDA.fill(A[n], num_points)
    Bₙ = CUDA.fill(B[n], num_points)
    Cₙ = CUDA.fill(C[n+1], num_points)
    cₙ = CUDA.fill(c[n], num_points)

    # calculate and return the next recurrence
    return ((Aₙ .* X + Bₙ) .* bn1 - Cₙ .* bn2 + cₙ)
end

# helper

function checkandconvert(x::AbstractVector)
    if eltype(x) == Float64
        return Float32.(x)
    else
        return x
    end
end