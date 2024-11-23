import RecurrenceRelationships: forwardrecurrence_next
import Base: size, show, string

mutable struct FixedRecurrenceArray{T, N, ZZ, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector} <: AbstractArray{T, N}
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

const FixedRecurrenceVector{T, A<:AbstractVector, B<:AbstractVector, C<:AbstractVector} = FixedRecurrenceArray{T, 1, T, A, B, C}
FixedRecurrenceArray(z, A, B, C, data::Array{T,N}, datasize, p0, p1) where {T,N} = FixedRecurrenceArray{T,N,typeof(z),typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, datasize, p0, p1, T[])


function initiateforwardrecurrence(N, A, B, C, x, μ)
    T = promote_type(eltype(A), eltype(B), eltype(C), typeof(x))
    p0 = convert(T, μ)
    N == 0 && return zero(T), p0
    p1 = convert(T, muladd(A[1],x,B[1])*p0)
    @inbounds for n = 2:N
        p1,p0 = forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
    end
    p0,p1
end

function FixedRecurrenceArray(z::Number, (A,B,C), data::AbstractVector{T}) where T
    N = length(data)
    p0, p1 = initiateforwardrecurrence(N, A, B, C, z, one(z))
    if iszero(p1)
        p1 = one(p1) # avoid degeneracy in recurrence. Probably needs more thought
    end
    FixedRecurrenceVector{T,typeof(A),typeof(B),typeof(C)}(z, A, B, C, data, size(data), T[p0], T[p1], T[])
end

size(R::FixedRecurrenceVector) = (size(R.data, 1))

function show(io::IO, ::MIME"text/plain", a::FixedRecurrenceArray)
    println(io, string(typeof(x)) * ":")
    println(io, x.data)
end