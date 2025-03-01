export ForwardInplace

# struct

struct ForwardInplace{T,Coefs<:AbstractVector,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector,XX<:AbstractVector}
    c::Coefs
    A::AA
    B::BB
    C::CC
    z::XX
    data::Array{T}
    f_z::Array{T}
    p0::T
end

# constructors

function ForwardInplace(c::AbstractVector, (A, B, C), z::AbstractVector,
    populate::Function=forwardvec_inplace!)

    num_coeffs = length(c)
    num_points = length(z)

    T = promote_type(eltype(c), eltype(z))

    num_coeffs == 0 && return zero(T)

    # copy the initialisation to a struct
    M = ForwardInplace(convert(AbstractVector{T}, c), A, B, C, 
        convert(AbstractVector{T}, z), Base.copy(z), zeros(T, num_points), one(T))

    # calculate and populate the data using forward_inplace
    populate(M.f_z, M.data, M.c, M.A, M.B, M.C)

    return M
end

ForwardInplace(c::Number, (A, B, C), z, p) = ForwardInplace([c], (A, B, C), z, p)
ForwardInplace(c, (A, B, C), z::Number, p) = ForwardInplace(c, (A, B, C), [z], p)

# display

function show(io::IO, ::MIME"text/plain", M::ForwardInplace)
    s = size(M)
    println(
        io,
        string(s[1]) * "×" * (length(s) > 1 ? string(s[2]) : string(1)) * " " *
        string(typeof(M)) * ":"
    )
    show(io, MIME"text/plain"(), M.f_z)
end


# serial population

function forwardvec_inplace!(f_z::AbstractVector, z::AbstractVector, c::AbstractVector, A, B, C)
    T = eltype(z)

    p0 = convert(T, one(eltype(z)))

    @inbounds for j in axes(z, 1)
        f_z[j] = forward_inplace(c, A, B, C, z[j], p0, muladd(A[1], z[j], B[1]) * p0)
    end
end

function forward_inplace(c::AbstractVector, A, B, C, z, p0, p1)
    num_coeffs = length(c)
    f_z = p0 * c[1] + p1 * c[2]

    @inbounds for i in 2:num_coeffs-1
        p1, p0 = muladd(muladd(A[i], z, B[i]), p1, -C[i]*p0), p1
        f_z += p1*c[i+1]
    end

    return f_z
end