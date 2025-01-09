import Base: size, show, getindex;
import RecurrenceRelationships: forwardrecurrence_next, forwardrecurrence_partial!;

export ThreadedFixedRecurrenceArray;

# constructor

function ThreadedFixedRecurrenceArray(z::AbstractVector, (A, B, C), input_data::AbstractMatrix{T}, n::Integer) where {T}
    return FixedRecurrenceArray(z, (A, B, C), input_data, n, multithreadedforwardrecurrence!)
end

# threaded population for evaluation of multiple points

function multithreadedforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), n::Integer) where {T,N}
    
    Threads.@threads for j in axes(z, 1)
        zⱼ = z[j]
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, zⱼ, start_index:n)
    end
end