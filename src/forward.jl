import Base: copy, size, show, string, getindex
import CUDA
import Distributed: workers, map, fetch
import DistributedData: save_at, get_val_from

import RecurrenceRelationships: forwardrecurrence_partial!, forwardrecurrence_next

export FixedRecurrenceArray, ThreadedRecurrenceArray, PartitionedRecurrenceArray,
    GPURecurrenceArray

# constructors

FixedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer) = 
    Forward(z, (A, B, C), n, Base.zeros(eltype(z), 1, length(z)))

FixedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer, input_data::AbstractMatrix{T}) where T = 
    Forward(z, (A, B, C), n, input_data)

ThreadedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer, dims::Val{1}) = 
    ForwardTransposed(z, (A, B, C), n, Base.zeros(eltype(z), 1, length(z)), rowthreadedrecurrence!)

ThreadedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer, input_data::AbstractMatrix{T}, dims::Val{1}) where T = 
    ForwardTransposed(z, (A, B, C), n, input_data, rowthreadedrecurrence!)

ThreadedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer, dims::Val{2}) = 
    Forward(z, (A, B, C), n, Base.zeros(eltype(z), 1, length(z)), columnthreadedrecurrence!)

ThreadedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer, input_data::AbstractMatrix{T}, dims::Val{2}) where T = 
    Forward(z, (A, B, C), n, input_data, columnthreadedrecurrence!)

GPURecurrenceArray(z::AbstractVector, (A, B, C), n::Integer) = 
    GPUForward(z, (A, B, C), n, Base.zeros(eltype(z), 1, length(z)))

GPURecurrenceArray(z::AbstractVector, (A, B, C), n::Integer, input_data::AbstractMatrix{T}) where T = 
    GPUForward(z, (A, B, C), n, input_data)

function Forward(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix{T}, populate!::Function=defaultforwardrecurrence!) where T

    @assert n >= 2

    N, M = size(input_data)

    # allocate a fixed size output matrix
    output_data = similar(input_data, n, M)

    if N < 2
        @. output_data[1, :] = Base.one(T)
        @. output_data[2, :] = (A[1] * z + B[1]) * Base.one(T)
        N = 2
    else
        # copy the initial data to the output
        output_data[axes(input_data)...] = input_data
    end

    # calculate and populate recurrence
    populate!(N, output_data, z, (A, B, C), n)

    return output_data
end

# Transpose the working layout to better suit the column-wise array access in Julia

function ForwardTransposed(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix{T}, populate!::Function=rowthreadedrecurrence!) where T

    @assert n >= 2

    N, M = size(input_data)

    output_data = similar(input_data, M, n)

    if N < 2
        @. output_data[:, 1] = Base.one(T)
        @. output_data[:, 2] = (A[1] * z + B[1]) * Base.one(T)
        N = 2
    else
        input_data_transposed = transpose(input_data)
        output_data[axes(input_data_transposed)...] = input_data_transposed
    end

    populate!(N, output_data, z, (A, B, C), n)

    return transpose(output_data)
end

function GPUForward(z::AbstractVector, (A, B, C), n::Integer, input_data::AbstractMatrix{RawT}) where RawT

    # enforce Float32/ComplexF32
    z = checkandconvert(z)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    input_data = checkandconvert(input_data)

    @assert n >= 2

    T = eltype(input_data)
    N, M = size(input_data)

    # copy the data onto the GPU
    gpu_z, gpu_A, gpu_B, gpu_C = CuArray(z), CuArray(A), CuArray(B), CuArray(C)

    # allocate a fixed size output matrix on the GPU
    gpu_output_data = CuArray{T}(undef, (M, n))

    if N < 2
        gpu_output_data[:, 1] .= CUDA.one(T)
        gpu_output_data[:, 2] .= (view(gpu_A, 1) .* gpu_z .+ view(gpu_B, 1)) .* CUDA.one(T)
        N = 2
    else
        gpu_input_data = CuArray(transpose(input_data))
        gpu_output_data[axes(gpu_input_data)...] = gpu_input_data
    end

    # initialise views for forward computation
    gpu_p0 = view(gpu_output_data, :, N-1)
    gpu_p1 = view(gpu_output_data, :, N)
    
    # populate result
    @inbounds for i = N:n-1
        gpu_next = view(gpu_output_data, :, i+1)
        gpu_next .= (view(gpu_A, i) .* gpu_z .+ view(gpu_B, i)) .* gpu_p1 .- view(gpu_C, i) .* gpu_p0

        gpu_p0, gpu_p1 = gpu_p1, gpu_next
    end

    return transpose(gpu_output_data)
end

# Float32/ComplexF32 helper function

function checkandconvert(x)
    if eltype(x) <: Complex && eltype(x) != ComplexF32
        @warn "Converting input vector(s) to ComplexF32 for improved performance..." x=x maxlog=1
        return ComplexF32.(x)
    elseif eltype(x) != Float32
        @warn "Converting input vector(s) to Float32 for improved performance..." x=x maxlog=1
        return Float32.(x)
    else
        return x
    end
end

# default serial population

function defaultforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z::AbstractVector, (A, B, C), n::Integer) where {T,N}

    @inbounds for j = axes(z, 1)
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, z[j], start_index:n)
    end
end

# column-wise threaded population

function columnthreadedrecurrence!(start_index::Integer, output_data::Array{T,N},
    z::AbstractVector, (A, B, C), n::Integer) where {T,N}

    @inbounds Threads.@threads for j in axes(z, 1)
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, z[j], start_index:n)
    end
end

# row-wise threaded population

function rowthreadedrecurrence!(start_index::Integer, output_data::Array{T,N},
    z::AbstractVector, (A, B, C), n::Integer) where {T,N}

    @inbounds for i in start_index:n-1
        Threads.@threads for j in axes(z, 1)
            output_data[j, i+1] =
                forwardrecurrence_next(i, A, B, C, z[j], output_data[j, i-1], output_data[j, i])
        end
    end
end

# struct

mutable struct PartitionedRecurrenceArray
    workers::Vector # list of workers 
    partitions::Vector  # list of indexes partitioned on workers
    N::Integer # number of recurrences
    M::Integer # number of points
end


function PartitionedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix{T}, workers::Vector=workers()) where T

    @assert n >= 2

    M = length(z)

    num_workers = length(workers)

    # ensure workers are available to distribute to
    @assert workers != [1]
    @assert num_workers >= 1

    # partition the vectors uniformly based on the number of workers
    num_vectors_per_worker = Int(ceil(M / num_workers))
    partitions =
        [(1:M)[i:min(i + num_vectors_per_worker - 1, end)] for i in 1:num_vectors_per_worker:M]

    # copy the data onto the workers (no quote to evaluate on main process worker local data to send)
    # sync
    map(
        fetch,
        # async
        [save_at(
            worker,
            :_LOCAL_DATA,
            Dict(
                "z" => z[partitions[worker_index]],
                "A_B_C" => (A, B, C),
                "N" => n,
                "input_data" => input_data[:, partitions[worker_index]]
            )
        ) for (worker_index, worker) in enumerate(workers)]
    )

    # column-wise distributed population
    # sync
    map(fetch,
        # async
        [save_at(
            worker,
            :LOCAL_Fixed_Recurrence_Array,
            # quote to evaluate locally on the worker process
            :(FixedRecurrenceArray(
                _LOCAL_DATA["z"],
                _LOCAL_DATA["A_B_C"],
                _LOCAL_DATA["N"],
                _LOCAL_DATA["input_data"],
            ))
        ) for worker in workers]
    )

    return PartitionedRecurrenceArray{T,typeof(A),typeof(B),typeof(C)}(A, B, C, workers, partitions, n, M)
end

# properties and access

size(K::PartitionedRecurrenceArray) = (K.N, K.M)
copy(K::PartitionedRecurrenceArray) = K

# display

function show(io::IO, ::MIME"text/plain", K::PartitionedRecurrenceArray)
    s = size(K)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(K)) * ": no preview available on partitioned array."
    )
end

# partitioned indexing

function getindex(K::PartitionedRecurrenceArray, i, j)

    i, j = (convert_to_range(i, K.N), convert_to_range(j, K.M))

    workers, local_ranges = global_to_local(K, j)

    result = zeros(T, (length(i), length(j)))

    column_count = 1

    for (worker, local_range) in zip(workers, local_ranges)
        # copy the local ranges
        fetch(save_at(worker, :LOCAL_i, i))
        fetch(save_at(worker, :LOCAL_j, local_range))

        local_values = get_val_from(worker, :(LOCAL_Fixed_Recurrence_Array[LOCAL_i, LOCAL_j]))

        result[1:length(i), column_count:column_count+length(local_range)-1] = local_values

        column_count += length(local_range)
    end

    return result
end

function global_to_local(K::PartitionedRecurrenceArray, column_range::AbstractUnitRange)
    workers, local_ranges = [], []

    for (worker, partition) in zip(K.workers, K.partitions)
        intersection = intersect(column_range, partition)

        if !isempty(intersection)
            # calculate the local column range within the partition
            local_range = (intersection .- first(partition) .+ 1)

            push!(workers, worker)
            push!(local_ranges, local_range)
        end
    end

    return (workers, local_ranges)
end

function convert_to_range(input::AbstractUnitRange, limit)
    if last(input) > limit || first(input) < 1
        throw(BoundsError())
    end
    return input
end

function convert_to_range(input::Number, limit)
    if input > limit || input < 1
        throw(BoundsError())
    end
    return input:input
end

function convert_to_range(_::Colon, limit)
    return 1:limit
end
