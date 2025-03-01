import RecurrenceRelationships: forwardrecurrence_partial!, forwardrecurrence_next
import Base: size, show, string, tail, getindex

using CUDA, Distributed, DistributedData

export FixedRecurrenceArray, 
    ThreadedRecurrenceArray, 
    PartitionedRecurrenceArray, 
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

#TODO: Fix this in line with the actual code, so that comparisons are possible.

function FixedRecurrenceArray(z::Number, (A, B, C), input_data::AbstractVector{T},
    n::Integer, populate::Function=defaultforwardrecurrence!) where {T}
    
    N = length(input_data)

    # allocate a fixed size output array
    output_data = similar(input_data, n)

    # copy the initial data to the output
    output_data[axes(input_data)...] = input_data

    # calculate and populate recurrence
    populate(N, output_data, z, (A, B, C), n)

    return FixedRecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, output_data, n)
end

function FixedRecurrenceArray(z::AbstractVector, (A, B, C), input_data::AbstractMatrix{T},
    n::Integer, populate::Function=defaultforwardrecurrence!) where {T}

    M, N = size(input_data)

    # allocate a fixed size output matrix
    output_data = similar(input_data, n, N)

    # copy the initial data to the output
    output_data[axes(input_data)...] = input_data

    # calculate and populate recurrence
    populate(M, output_data, z, (A, B, C), n)

    return FixedRecurrenceMatrix{T,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, output_data, n)
end

# dim 1: rows, dim 2: columns

function ThreadedRecurrenceArray(z::AbstractVector, (A, B, C), 
    input_data::AbstractMatrix{T}, n::Integer, dims::Integer=1) where {T}

    @assert dims == 1 || dims == 2 "dimension must be either 1 or 2."

    if dims == 1
        return FixedRecurrenceArray(z, (A, B, C), input_data, n, rowthreadedrecurrence!)
    elseif dims == 2
        return FixedRecurrenceArray(z, (A, B, C), input_data, n, columnthreadedrecurrence!)
    end
end

function PartitionedRecurrenceArray(z::AbstractVector, (A, B, C),
    input_data::AbstractMatrix{T}, n::Integer,
    workers::Vector=workers()) where {T}

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
                    "input_data" => input_data[:, partitions[worker_index]],
                    "N" => n
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
                _LOCAL_DATA["input_data"],
                _LOCAL_DATA["N"]
            ))
        ) for worker in workers]
    )

    return PartitionedRecurrenceArray{T,typeof(A),typeof(B),typeof(C)}(A, B, C, workers, partitions, n, M)
end

function GPURecurrenceArray(z::AbstractVector, (A, B, C),
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

# Float32/ComplexF32 helper function

function checkandconvert(x)
    if eltype(x) == Float64
        return Float32.(x)
    elseif eltype(x) == ComplexF64
        return ComplexF32.(x)
    else
        return x
    end
end

# properties and access

size(K::FixedRecurrenceVector) = (K.n,)
size(K::FixedRecurrenceMatrix) = (K.n, size(K.data)[2])
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
    
        local_values =  get_val_from(worker, :(LOCAL_Fixed_Recurrence_Array[LOCAL_i, LOCAL_j]))

        result[1:length(i), column_count : column_count + length(local_range) - 1] = local_values

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
            output_data[i+1, j] = 
                forwardrecurrence_next(i, A, B, C, z[j], output_data[i-1, j], output_data[i, j])
        end
    end
end

# row-wise GPU population

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

function gpuforwardrecurrence_next(n::Integer, A, B, C, z::CuArray, p0::CuArray,
    p1::CuArray, num_points::Integer)

    # construct vectors
    Aₙ = CUDA.fill(A[n], num_points)
    Bₙ = CUDA.fill(B[n], num_points)
    Cₙ = CUDA.fill(C[n], num_points)

    # calculate and return the next recurrence
    return ((Aₙ .* z + Bₙ) .* p1 - Cₙ .* p0)
end