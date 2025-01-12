import RecurrenceRelationships: clenshaw, clenshaw!
import BandedMatrices: AbstractBandedMatrix, bandwidth
import Base: copy, size, show, getindex

export FixedClenshaw

# struct

struct FixedClenshaw{T,Coefs<:AbstractVector,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector,Jac<:Union{AbstractMatrix, AbstractVector}} <: AbstractBandedMatrix{T}
    c::Coefs
    A::AA
    B::BB
    C::CC
    X::Jac
    data::Array{T}
    p0::T
end


# constructors

function FixedClenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector,
    C::AbstractVector, X::AbstractMatrix)

    T = promote_type(eltype(c), eltype(X))

    M, N = size(X)

    # allocate a fixed size output array
    output_data = zeros(M, N)

    # copy the initialisation to a struct
    M = FixedClenshaw(convert(AbstractVector{T}, c), A, B, C, convert(AbstractMatrix{T}, X), output_data, one(T))

    # calculate and populate the data using Clenshaw's
    matrixclenshaw!(output_data, M)

    return M
end

function FixedClenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector,
    C::AbstractVector, X::AbstractVector)

    T = promote_type(eltype(c), eltype(X))

    # copy the initialisation to a struct
    M = FixedClenshaw(convert(AbstractVector{T}, c), A, B, C, convert(AbstractVector{T}, X), copy(X), one(T))

    # calculate and populate the data using Clenshaw's
    clenshaw!(M.data, M.c, M.A, M.B, M.C)

    return M
end

FixedClenshaw(c::Number, A, B, C, X, p) = FixedClenshaw([c], A, B, C, X, p)
FixedClenshaw(c, A, B, C, x::Number, p) = FixedClenshaw(c, A, B, C, [x], p)

# properties and access

copy(M::FixedClenshaw) = M # immutable entries
size(M::FixedClenshaw) = size(M.data)
axes(M::FixedClenshaw) = axes(M.data)
bandwidths(M::FixedClenshaw) = (length(M.c) - 1, length(M.c) - 1)
getindex(M::FixedClenshaw, index...) = M.data[index...]

# display

function show(io::IO, ::MIME"text/plain", M::FixedClenshaw)
    s = size(M)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(M)) * ":"
    )
    show(io, MIME"text/plain"(), M.data)
end


# population

function matrixclenshaw!(output_data::Array{T}, M::FixedClenshaw) where {T}
    n, m = size(M)

    b = bandwidth(M, 1)

    for i in 1:n
        kr = i:i
        for j in 1:m
            jkr = max(1, min(j, first(kr)) - b ÷ 2):min(m, max(j, last(kr)) + b ÷ 2)

            # relationship between jkr and kr, jr
            kr2, j2 = kr .- first(jkr) .+ 1, j - first(jkr) + 1

            f = [zeros(j2 - 1); one(T); zeros(length(jkr) - j2)]
            output_data[kr, j] = (M.p0 * clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr], f)[kr2])
        end
    end
end