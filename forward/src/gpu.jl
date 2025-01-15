using CUDA

export GPUFixedRecurrenceArray

# constructor

function GPUFixedRecurrenceArray(z::AbstractVector, (A, B, C),
    input_data::AbstractMatrix{T}, n::Integer) where {T}

    if (eltype(z) == Float64 || eltype(z) == ComplexF64 || eltype(A) == Float64 ||
        eltype(B) == Float64 || eltype(C) == Float64 || T == Float64 || T == ComplexF64)
        @warn "Converting input vector(s) to Float32 for improved performance..."
    end

    # enforce Float32
    z = checkandconvert(z)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    input_data = checkandconvert(input_data)

    return FixedRecurrenceArray(z, (A, B, C), input_data, n, gpuforwardrecurrence!)
end

# GPU population for evaluation of multiple points
function gpuforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), num_recurrences::Integer) where {T,N}

    num_points = length(z)

    # copy the data to the GPU
    gpu_z = CuArray(z)

    # initialise arrays for forward computation
    gpu_p0, gpu_p1 = CuArray(output_data[start_index-1, :]), CuArray(output_data[start_index, :])

    # initialise result storage
    gpu_result = CuArray(output_data)

    # populate result
    @inbounds for i = start_index:num_recurrences-1
        gpu_p1, gpu_p0 = gpuforwardrecurrence_next(i, A, B, C, gpu_z, gpu_p0, gpu_p1, num_points), gpu_p1
        view(gpu_result, i + 1, :) .= gpu_p1
    end

    # copy result to memory
    copyto!(output_data, gpu_result)
end

# forward
function gpuforwardrecurrence_next(n::Integer, A, B, C, z::CuArray, p0::CuArray,
    p1::CuArray, num_points::Integer)

    # construct vectors
    Aₙ = CUDA.fill(A[n], num_points)
    Bₙ = CUDA.fill(B[n], num_points)
    Cₙ = CUDA.fill(C[n], num_points)

    # calculate and return the next recurrence
    return ((Aₙ .* z + Bₙ) .* p1 - Cₙ .* p0)
end

# helper

function checkandconvert(x)
    if eltype(x) == Float64
        return Float32.(x)
    elseif eltype(x) == ComplexF64
        return ComplexF32.(x)
    else
        return x
    end
end