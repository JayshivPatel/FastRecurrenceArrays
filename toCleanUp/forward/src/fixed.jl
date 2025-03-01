import RecurrenceRelationships: forwardrecurrence_partial!
import RecurrenceRelationshipArrays: initiateforwardrecurrence
import Base: size, show, string, tail, getindex

export FixedRecurrenceArray

# struct
mutable struct FixedRecurrenceArray{T,N,ZZ,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector} <: AbstractArray{T,N}
    z::ZZ # evaluation point
    A::AA # 3-term-recurrence A
    B::BB # 3-term-recurrence B
    C::CC # 3-term-recurrence C
    data::Array{T,N}
    n::Integer # number of recurrences
end

# types

FixedRecurrenceVector{T,A<:AbstractVector,B<:AbstractVector,C<:AbstractVector} =
    FixedRecurrenceArray{T,1,T,A,B,C}
FixedRecurrenceMatrix{T,Z<:AbstractVector,A<:AbstractVector,B<:AbstractVector,C<:AbstractVector} =
    FixedRecurrenceArray{T,2,Z,A,B,C}

FixedRecurrenceArray(z, A, B, C, data::Array{T,N}, n) where {T,N} =
    FixedRecurrenceArray{T,N,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, n)

# constructors

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

# properties and access

size(K::FixedRecurrenceVector) = (K.n,)
size(K::FixedRecurrenceMatrix) = (K.n, size(K.data)[2])
copy(K::FixedRecurrenceArray) = K # immutable entries
getindex(K::FixedRecurrenceArray, index...) = K.data[index...]

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

# population

function defaultforwardrecurrence!(start_index::Integer, output_data::Array{T,N},
    z, (A, B, C), n::Integer) where {T,N}

    for j = axes(z, 1)
        zⱼ = z[j]
        forwardrecurrence_partial!(view(output_data, :, j), A, B, C, zⱼ, start_index:n)
    end
end