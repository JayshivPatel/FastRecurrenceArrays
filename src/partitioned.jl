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

    # distribute the population
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

    return PartitionedFixedRecurrenceArray{T,typeof(A),typeof(B),typeof(C)}(A, B, C, workers, partitions, n, M)
end

# properties and access

size(K::PartitionedFixedRecurrenceArray) = (K.N, K.M)
copy(K::PartitionedFixedRecurrenceArray) = K # immutable entries

function getindex(K::PartitionedFixedRecurrenceArray{T}, i, j) where {T}

    i, j = indexes_to_ranges(K, i, j)

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

function indexes_to_ranges(K::PartitionedFixedRecurrenceArray, i, j)
    return (convert_to_range(i, K.N), convert_to_range(j, K.M))
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

# display

function show(io::IO, ::MIME"text/plain", K::PartitionedFixedRecurrenceArray)
    s = size(K)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(K)) * ": no preview available on partitioned array."
    )
end