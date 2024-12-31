import RecurrenceRelationships: forwardrecurrence_next, forwardrecurrence_partial!;
import RecurrenceRelationshipArrays: initiateforwardrecurrence;
import Base: size, show, string, tail, getindex;

using Distributed, DistributedData;

export PartitionedFixedRecurrenceArray;

# struct
mutable struct PartitionedFixedRecurrenceArray{T,AA<:AbstractVector,
    BB<:AbstractVector,CC<:AbstractVector} <: AbstractArray{T,2}

    A::AA # 3-term-recurrence A
    B::BB # 3-term-recurrence B
    C::CC # 3-term-recurrence C
    workers::Vector # list of workers 
    partitions::Vector  # list of indexes partitioned on workers
    N::Integer # number of recurrences
    M::Integer # number of vectors
end

# constructor

function PartitionedFixedRecurrenceArray(z::AbstractVector, (A, B, C),
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

    # distribute the population on the available workers
    @sync for (worker_index, worker) in enumerate(workers)
        @async begin
            # copy the local points and input data
            fetch(save_at(
                worker,
                :_LOCAL_z,
                # no quote to only send the local points
                z[partitions[worker_index]]
            ))

            fetch(save_at(
                worker,
                :_LOCAL_input_data,
                # no quote to only send the local input data
                input_data[:, partitions[worker_index]]
            ))

            # copy the coefficients and number of recurrences
            fetch(save_at(
                worker,
                :_LOCAL_A_B_C,
                (A, B, C)
            ))

            fetch(save_at(
                worker,
                :_LOCAL_N,
                n
            ))

            # async
            # populate locally on the worker
            save_at(
                worker,
                :LOCAL_Fixed_Recurrence_Array,
                # quote to evaluate on the worker
                :(FixedRecurrenceArray(_LOCAL_z, _LOCAL_A_B_C, _LOCAL_input_data, _LOCAL_N))
            )
        end
    end

    return PartitionedFixedRecurrenceArray{T,typeof(A),typeof(B),typeof(C)}(A, B, C, workers, partitions, n, M)
end

# properties and access

size(K::PartitionedFixedRecurrenceArray) = (K.N, K.M)
copy(K::PartitionedFixedRecurrenceArray) = K # immutable entries


function getindex(K::PartitionedFixedRecurrenceArray, i, j::Int)
    worker, local_j = global_to_local(K, j)
    
    # copy the local indexes
    save_at(worker, :LOCAL_i, i)
    save_at(worker, :LOCAL_j, local_j)
    
    return get_val_from(worker, :(LOCAL_Fixed_Recurrence_Array[LOCAL_i, LOCAL_j]))
end

function getindex(K::PartitionedFixedRecurrenceArray{T}, i, j::AbstractUnitRange) where {T}

    workers, local_ranges = global_to_local(K, j)

    if i == Colon()
        result = zeros(T, (K.N, length(j)))
    else
        result = zeros(T, (length(i), length(j)))
    end

    column = 1

    for (worker, local_range) in zip(workers, local_ranges)
        # copy the local ranges
        save_at(worker, :LOCAL_i, i)
        save_at(worker, :LOCAL_j, local_range)
    
        local_values =  get_val_from(worker, :(LOCAL_Fixed_Recurrence_Array[LOCAL_i, LOCAL_j]))

        result[i, column : column + length(local_range) - 1] = local_values
    end
    
    return result
end

function global_to_local(K::PartitionedFixedRecurrenceArray, column_index::Int)
    for (worker, partition) in zip(K.workers, K.partitions)
        if column_index in partition
            # calculate the local column index within the partition
            j_local = column_index - partition[1] + 1
            
            return (worker, j_local)
        end
    end
end

function global_to_local(K::PartitionedFixedRecurrenceArray, column_range::AbstractUnitRange)
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

# display

function show(io::IO, ::MIME"text/plain", K::PartitionedFixedRecurrenceArray)
    s = size(K)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(K)) * ": no preview available on partitioned array."
    )
end