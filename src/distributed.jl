import Base: size, show, getindex;
import RecurrenceRelationships: forwardrecurrence_next, forwardrecurrence_partial!;

export DistributedFixedRecurrenceArray;

using Distributed;

# constructors

function DistributedFixedRecurrenceArray(z::AbstractVector, (A, B, C), input_data::AbstractMatrix{T}, n::Integer) where {T}
    return FixedRecurrenceArray(z, (A, B, C), input_data, n, distributedforwardrecurrence!)
end

# distributed population for evaluation of multiple points

function distributedforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), n::Integer) where {T,N}
    
    @distributed for j in axes(z, 1)
        zⱼ = z[j]
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, zⱼ, start_index:n)
    end
end