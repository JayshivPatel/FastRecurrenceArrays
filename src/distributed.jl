import RecurrenceRelationships: forwardrecurrence_next, forwardrecurrence_partial!;
import RecurrenceRelationshipArrays: initiateforwardrecurrence;
import Base: size, show, string, tail, getindex;

using Distributed, DistributedData;

export DistributedFixedRecurrenceArray;

# struct
mutable struct DistributedFixedRecurrenceArray{T,AA<:AbstractVector,
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

function DistributedFixedRecurrenceArray(z::AbstractVector, (A, B, C),
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
            local_input_data = input_data[:, partitions[worker_index]]
            local_z = z[partitions[worker_index]]
            save_at(
                worker,
                :DistributedFixedRecurrenceArray_LOCAL,
                FixedRecurrenceArray(local_z, (A, B, C), local_input_data, n)
            )
        end
    end

    return DistributedFixedRecurrenceArray{T,typeof(A),typeof(B),typeof(C)}(A, B, C, workers, partitions, n, M)
end

# properties and access

size(K::DistributedFixedRecurrenceArray) = (K.N, K.M)
copy(K::DistributedFixedRecurrenceArray) = K # immutable entries

function getindex(K::DistributedFixedRecurrenceArray, i::Union{Int,UnitRange}, 
    j::Union{Int,UnitRange})

    worker, j_local = global_to_local(K, j)
    
    save_at(worker, :i_LOCAL, i)
    save_at(worker, :j_LOCAL, j_local)
    
    return get_val_from(worker, :(DistributedFixedRecurrenceArray_LOCAL[i_LOCAL, j_LOCAL]))
end

function global_to_local(K::DistributedFixedRecurrenceArray, column_index::Int)
    for (worker, partition) in zip(K.workers, K.partitions)
        if column_index in partition
            # calculate the local column index within the partition
            j_local = column_index - partition[1] + 1
            
            return (worker, j_local)
        end
    end
end

# display

function show(io::IO, ::MIME"text/plain", K::DistributedFixedRecurrenceArray)
    s = size(K)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(K)) * ": no preview available on distributed array."
    )
end