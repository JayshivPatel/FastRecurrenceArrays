import Base: size, show, string, tail, getindex
import CUDA: CuArray, fill
import Distributed: workers, map, fetch
import DistributedData: save_at, get_val_from

import RecurrenceRelationships: forwardrecurrence_partial!, forwardrecurrence_next

export FixedRecurrenceArray, ThreadedRecurrenceArray, PartitionedRecurrenceArray,
    GPURecurrenceArray

# structs

mutable struct FixedRecurrenceArray{T,N,ZZ,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector} <: AbstractArray{T,N}
    z::ZZ # evaluation point
    A::AA # 3-term-recurrence A
    B::BB # 3-term-recurrence B
    C::CC # 3-term-recurrence C
    data::Array{T,N}
    n::Integer # number of recurrences
end

mutable struct PartitionedRecurrenceArray{T,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector} <: AbstractArray{T,2}
    A::AA
    B::BB
    C::CC
    workers::Vector # list of workers 
    partitions::Vector  # list of indexes partitioned on workers
    N::Integer # number of recurrences
    M::Integer # number of vectors
end

# types

FixedRecurrenceVector{T,A<:AbstractVector,B<:AbstractVector,C<:AbstractVector} =
    FixedRecurrenceArray{T,1,T,A,B,C}
FixedRecurrenceMatrix{T,Z<:AbstractVector,A<:AbstractVector,B<:AbstractVector,C<:AbstractVector} =
    FixedRecurrenceArray{T,2,Z,A,B,C}

FixedRecurrenceArray(z, A, B, C, data::Array{T,N}, n) where {T,N} =
    FixedRecurrenceArray{T,N,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, n)

# constructors

function FixedRecurrenceArray(z::Number, (A, B, C), n::Integer,
    input_data::AbstractVector{T}=Base.zeros(typeof(z), 1), populate::Function=defaultforwardrecurrence!) where {T}

    @assert n >= 2

    N = length(input_data)

    # allocate a fixed size output array
    output_data = similar(input_data, n)

    if N < 2
        p0 = convert(T, one(z))
        p1 = convert(T, muladd(A[1], z, B[1]) * p0)
        output_data[1] = p0
        output_data[2] = p1

        N = 2
    else
        # copy the initial data to the output
        output_data[axes(input_data)...] = input_data
    end

    # calculate and populate recurrence
    populate(N, output_data, z, (A, B, C), n)

    return FixedRecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, output_data, n)
end

function FixedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix{T}=Base.zeros(eltype(z), 1, length(z)), populate::Function=defaultforwardrecurrence!) where {T}

    @assert n >= 2

    N, M = size(input_data)

    # allocate a fixed size output matrix
    output_data = similar(input_data, n, M)

    if N < 2
        p0 = Vector{T}(undef, M)
        p1 = Vector{T}(undef, M)

        for j = axes(z, 1)
            p0[j] = convert(T, one(z[j]))
            p1[j] = convert(T, muladd(A[1], z[j], B[1]) * p0[j])
        end

        output_data[1, :] .= p0
        output_data[2, :] .= p1

        N = 2
    else
        # copy the initial data to the output
        output_data[axes(input_data)...] = input_data
    end

    # calculate and populate recurrence
    populate(N, output_data, z, (A, B, C), n)

    return FixedRecurrenceMatrix{T,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, output_data, n)
end

function TransposedFixedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix{T}=Base.zeros(eltype(z), 1, length(z)), populate::Function=defaultforwardrecurrence!) where {T}

    @assert n >= 2

    N, M = size(input_data)

    # allocate a fixed size output matrix (transposed)
    output_data = similar(input_data, M, n)

    input_data_transposed = permutedims(input_data)

    if N < 2
        p0 = Vector{T}(undef, M)
        p1 = Vector{T}(undef, M)

        for j = axes(z, 1)
            p0[j] = convert(T, one(z[j]))
            p1[j] = convert(T, muladd(A[1], z[j], B[1]) * p0[j])
        end

        output_data[:, 1] .= p0
        output_data[:, 2] .= p1

        N = 2
    else
        # copy the initial data to the output
        output_data[axes(input_data_transposed)...] = input_data_transposed
    end

    # calculate and populate recurrence
    populate(N, output_data, z, (A, B, C), n)

    return FixedRecurrenceMatrix{T,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, output_data, n)
end

# dim 1: rows, dim 2: columns

function ThreadedRecurrenceArray(z::AbstractVector, (A, B, C),
    n::Integer, dims::Integer=1, input_data::AbstractMatrix{T}=Base.zeros(eltype(z), 1, length(z))) where {T}

    @assert n >= 2
    @assert dims == 1 || dims == 2 "dimension must be either 1 or 2."

    if dims == 1
        return TransposedFixedRecurrenceArray(z, (A, B, C), n, input_data, rowthreadedrecurrence!)
    elseif dims == 2
        return FixedRecurrenceArray(z, (A, B, C), n, input_data, columnthreadedrecurrence!)
    end
end

function PartitionedRecurrenceArray(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix{T}=Base.zeros(eltype(z), 1, length(z)), workers::Vector=workers()) where {T}

    @assert n >= 2

    # get the number of vectors
    _, M = size(input_data)

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

function GPURecurrenceArray(z::AbstractVector, (A, B, C), n::Integer,
    input_data::AbstractMatrix=Base.zeros(Float32, (1, length(z))))

    # enforce Float32
    z = checkandconvert(z)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    input_data = checkandconvert(input_data)

    return TransposedFixedRecurrenceArray(z, (A, B, C), n, input_data, gpuforwardrecurrence!)
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

# properties and access

size(K::FixedRecurrenceVector) = (K.n,)
size(K::FixedRecurrenceMatrix) = size(K.data)
copy(K::FixedRecurrenceArray) = K # immutable entries
getindex(K::FixedRecurrenceArray, index...) = K.data[index...]

size(K::PartitionedRecurrenceArray) = (K.N, K.M)
copy(K::PartitionedRecurrenceArray) = K

# display

function show(io::IO, ::MIME"text/plain", K::FixedRecurrenceArray)
    s = size(K)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(K)) * ":"
    )
    show(io, MIME"text/plain"(), K.data)
end

function show(io::IO, ::MIME"text/plain", K::PartitionedRecurrenceArray)
    s = size(K)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(K)) * ": no preview available on partitioned array."
    )
end

# partitioned indexing

function getindex(K::PartitionedRecurrenceArray{T}, i, j) where {T}

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

function convert_to_range(input, limit)
    if isa(input, AbstractUnitRange)
        if last(input) > limit || first(input) < 1
            throw(BoundsError())
        end
        return input
    elseif isa(input, Number)
        if input > limit || input < 1
            throw(BoundsError())
        end
        return input:input
    elseif isa(input, Colon)
        return 1:limit
    end
end

# default serial population

function defaultforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), n::Integer) where {T,N}

    @inbounds for j = axes(z, 1)
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, z[j], start_index:n)
    end
end

# column-wise threaded population

function columnthreadedrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), n::Integer) where {T,N}

    @inbounds Threads.@threads for j in axes(z, 1)
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, z[j], start_index:n)
    end
end

# row-wise threaded population

function rowthreadedrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), n::Integer) where {T,N}

    @inbounds for i in start_index:n-1
        Threads.@threads for j in axes(z, 1)
            output_data[j, i+1] =
                forwardrecurrence_next(i, A, B, C, z[j], output_data[j, i-1], output_data[j, i])
        end
    end
end

# row-wise GPU population

function gpuforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), num_recurrences::Integer) where {T,N}

    num_points = length(z)

    # copy the data to the GPU
    gpu_z = CuArray(z)

    # initialise result storage
    gpu_result = CuArray(output_data)

    # initialise views for forward computation
    gpu_p0 = view(gpu_result, :, start_index-1)
    gpu_p1 = view(gpu_result, :, start_index)

    # populate result
    @inbounds for i = start_index:num_recurrences-1
        gpu_next = view(gpu_result, :, i+1)
        gpuforwardrecurrence_next!(gpu_next, A[i], B[i], C[i], gpu_z, gpu_p0, gpu_p1)

        gpu_p0, gpu_p1 = gpu_p1, gpu_next
    end

    # copy result to memory
    copyto!(output_data, gpu_result)
end

function gpuforwardrecurrence_next!(output::CuArray, A, B, C, z::CuArray, p0::CuArray, p1::CuArray)

    @. output = (A * z + B) * p1 - C * p0
end