import RecurrenceRelationships: clenshaw
import BandedMatrices: AbstractBandedMatrix, bandwidth

export FixedClenshaw
# struct

struct FixedClenshaw{T, Coefs<:AbstractVector, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector, Jac<:AbstractMatrix} <: AbstractBandedMatrix{T}
    c::Coefs
    A::AA
    B::BB
    C::CC
    X::Jac
    data::Array{T}
    p0::T
end

# types

FixedClenshaw(c::AbstractVector{T}, A::AbstractVector, B::AbstractVector, 
    C::AbstractVector, X::AbstractMatrix{T}, output_data::Array{T}, p0=one(T)) where T = 
    FixedClenshaw{T,typeof(c),typeof(A),typeof(B),typeof(C),typeof(X)}(c, A, B, C, X, output_data, p0)

# constructors

function FixedClenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, 
    C::AbstractVector, X::AbstractMatrix, populate::Function=defaultclenshaw!, p0...)

    T = promote_type(eltype(c), eltype(X))

    M, N = size(X)

    # allocate a fixed size output array
    output_data = zeros(M, N)

    # copy the initialisation to a struct
    matrix = FixedClenshaw(convert(AbstractVector{T}, c), A, B, C, convert(AbstractMatrix{T},X), output_data, p0...)

    # calculate and populate the data using Clenshaw's
    populate(output_data, matrix)

    return matrix
end

FixedClenshaw(c::Number, A, B, C, X, p) = FixedClenshaw([c], A, B, C, X, p)

# properties and access

copy(M::FixedClenshaw) = M # immutable entries
size(M::FixedClenshaw) = size(M.X)
axes(M::FixedClenshaw) = axes(M.X)
bandwidths(M::FixedClenshaw) = (length(M.c)-1,length(M.c)-1)
getindex(M::FixedClenshaw, index...) = M.data[index...]

# population

function defaultclenshaw!(output_data::Array{T}, M::FixedClenshaw) where {T} 
    n, m = size(M)

    b = bandwidth(M, 1)

    for i in 1:n
        kr = i:i
        for j in 1:m
            jkr = max(1,min(j,first(kr))-b÷2):min(m, max(j,last(kr))+b÷2)

            # relationship between jkr and kr, jr
            kr2,j2 = kr.-first(jkr).+1,j-first(jkr)+1
            
            f = [zeros(j2-1); one(T); zeros(length(jkr)-j2)]
            output_data[kr, j] = (M.p0 * clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr], f)[kr2])
        end
    end
end