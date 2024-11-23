import RecurrenceRelationships: forwardrecurrence_next, forwardrecurrence_partial!
import Base: size, show, string, tail

mutable struct FixedRecurrenceArray{T,N,ZZ,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector} <: AbstractArray{T,N}
    z::ZZ
    A::AA
    B::BB
    C::CC
    data::Array{T,N}
    datasize::NTuple{N,Int}
    p0::Vector{T} # stores p_{s-1} to determine when to switch to backward
    p1::Vector{T} # stores p_{s} to determine when to switch to backward
    u::Vector{T} # used for backsubstitution to store diagonal of U in LU
end

const FixedRecurrenceVector{T,A<:AbstractVector,B<:AbstractVector,C<:AbstractVector} = 
    FixedRecurrenceArray{T,1,T,A,B,C}
const FixedRecurrenceMatrix{T,Z<:AbstractVector,A<:AbstractVector,B<:AbstractVector,C<:AbstractVector} = 
    FixedRecurrenceArray{T,2,Z,A,B,C}

FixedRecurrenceArray(z, A, B, C, data::Array{T,N}, datasize, p0, p1) where {T,N} = 
    FixedRecurrenceArray{T,N,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, datasize, p0, p1, T[])

function initiateforwardrecurrence(N, A, B, C, x, μ)
    T = promote_type(eltype(A), eltype(B), eltype(C), typeof(x))
    p0 = convert(T, μ)
    N == 0 && return zero(T), p0
    p1 = convert(T, muladd(A[1], x, B[1]) * p0)
    @inbounds for n = 2:N
        p1, p0 = forwardrecurrence_next(n, A, B, C, x, p0, p1), p1
    end
    return p0, p1
end

function populateforwardrecurrence(K::FixedRecurrenceArray, n::Integer)
    v = K.datasize[1]

    if size(K.data, 2) > 1
        _growdata!(K, n, size(K.data, 2)...)
    else
        _growdata!(K, n)
    end

    A,B,C = K.A,K.B,K.C
    for j = axes(K.z,1)
        z = K.z[j]
        p0, p1 = K.p0[j], K.p1[j]
        k = v
        while k < n
            p1, p0 = forwardrecurrence_next(k, A, B, C, z, p0, p1), p1
            k += 1
        end
        K.p0[j], K.p1[j] = p0, p1

        forwardrecurrence_partial!(view(K.data,:,j), A, B, C, z, v:k)
    end

    K.datasize = (max(K.datasize[1],n), tail(K.datasize)...)
end

function FixedRecurrenceArray(z::Number, (A, B, C), data::AbstractVector{T}, n::Integer) where {T}
    N = length(data)
    p0, p1 = initiateforwardrecurrence(N, A, B, C, z, one(z))
    if iszero(p1)
        p1 = one(p1) # avoid degeneracy in recurrence. Probably needs more thought
    end
    K = FixedRecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), T[p0], T[p1], T[])
    populateforwardrecurrence(K, n)
    return K
end

function FixedRecurrenceArray(z::AbstractVector, (A, B, C), data::AbstractMatrix{T}, n::Integer) where {T}
    M, N = size(data)
    p0 = Vector{T}(undef, N)
    p1 = Vector{T}(undef, N)
    for j = axes(z, 1)
        p0[j], p1[j] = initiateforwardrecurrence(M, A, B, C, z[j], one(T))
        if iszero(p1[j])
            p1[j] = one(p1[j]) # avoid degeneracy in recurrence. Probably needs more thought
        end
    end
    K = FixedRecurrenceMatrix{T,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), p0, p1, T[])
    populateforwardrecurrence(K, n)
    return K
end

size(R::FixedRecurrenceVector) = (size(R.data, 1))
size(R::FixedRecurrenceMatrix) = (size(R.data, 1), size(R.data, 2))
copy(R::FixedRecurrenceArray) = R # immutable entries

function _growdata!(B::AbstractArray{<:Any,N}, nm::Vararg{Integer,N}) where N
    # increase size of array if necessary
    olddata = B.data
    νμ = size(olddata)
    nm = max.(νμ,nm)
    if νμ ≠ nm
        B.data = similar(B.data, nm...)
        B.data[axes(olddata)...] = olddata
    end
end

function show(io::IO, ::MIME"text/plain", x::FixedRecurrenceArray)
    println(
        io, 
        string(x.datasize[1]) * "×" * (length(x.datasize) > 1 ? string(x.datasize[2]) : string(1)) * " " * 
        string(typeof(x)) * ":"
    )
    show(io, MIME"text/plain"(), x.data)
end