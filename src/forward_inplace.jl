export ForwardInplace, ThreadedInplace, GPUInplace

# struct

struct ForwardInplace{T,Coefs<:AbstractVector,AA<:AbstractVector,BB<:AbstractVector,CC<:AbstractVector,XX<:AbstractVector}
    c::Coefs
    A::AA
    B::BB
    C::CC
    z::XX
    f_z::Array{T}
    p0::T
end

# constructors

function ForwardInplace(c::AbstractVector, (A, B, C), z::AbstractVector,
    input_data::AbstractMatrix=zeros(eltype(z), 1, length(z)), populate::Function=forwardvec_inplace!)

    num_coeffs = length(c)
    num_points = length(z)

    T = promote_type(eltype(c), eltype(z))

    num_coeffs == 0 && return zero(T)
    
    M, N = size(input_data)
    f_z = zeros(eltype(z), num_points)

    p0 = Vector{T}(undef, N)
    p1 = Vector{T}(undef, N)

    if M < 2
        for j = axes(z, 1)
            p0[j] = convert(T, one(z[j]))
            p1[j] = convert(T, muladd(A[1], z[j], B[1])*p0[j])
        end
        f_z += p0 * c[1] + p1 * c[2]
    else
        for i in 1:M
            f_z += input_data[i, :] * c[1]
        end
        p0 = input_data[end-1, :]
        p1 = input_data[end, :]
    end

    # calculate and populate f_z using forward_inplace
    populate(f_z, z, c[M:end], A, B, C, p0, p1)

    return ForwardInplace(c, A, B, C, z, f_z, one(T))
end

function ThreadedInplace(c::AbstractVector, (A, B, C), z::AbstractVector,
    input_data::AbstractMatrix=zeros(eltype(z), 1, length(z)))

    return ForwardInplace(c, (A, B, C), z, input_data, threaded_inplace!)
end

function GPUInplace(c::AbstractVector, (A, B, C), z::AbstractVector,
    input_data::AbstractMatrix=zeros(eltype(z), 1, length(z)))

    if (eltype(c) == Float64 || eltype(A) == Float64 || eltype(B) == Float64 ||
        eltype(C) == Float64 || eltype(z) == Float64)
        @warn "Converting input vector(s) to Float32 for improved performance..."
    end

    # enforce Float32
    c = checkandconvert(c)
    A = checkandconvert(A)
    B = checkandconvert(B)
    C = checkandconvert(C)
    z = checkandconvert(z)

    return ForwardInplace(c, (A, B, C), z, input_data, gpu_inplace!)
end

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

size(M::ForwardInplace) = size(M.f_z)

# serial population

function forwardvec_inplace!(f_z::AbstractVector, z::AbstractVector, c::AbstractVector, 
    A, B, C, p0::AbstractVector, p1::AbstractVector)

    @inbounds for j in axes(z, 1)
        f_z[j] += forward_inplace(c, A, B, C, z[j], p0[j], p1[j])
    end
end

# threaded population (column)

function threaded_inplace!(f_z::AbstractVector, z::AbstractVector, c::AbstractVector, 
    A, B, C, p0::AbstractVector, p1::AbstractVector)

    @inbounds Threads.@threads for j in axes(z, 1)
        f_z[j] += forward_inplace(c, A, B, C, z[j], p0[j], p1[j])
    end
end

# gpu population 
function gpu_inplace!(f_z::AbstractVector, z::AbstractVector, c::AbstractVector, 
    A, B, C, p0::AbstractVector, p1::AbstractVector)

    num_coeffs = length(c)
    num_points = length(z)

    @inbounds begin
        # copy the data to the GPU
        gpu_z = CuArray(z)
        gpu_f_z = CuArray(f_z)

        # initialise arrays for the clenshaw computation
        gpu_p0, gpu_p1 = CuArray(p0), CuArray(p1)

        num_coeffs == 1 && return Array(gpu_bn1)

        for n = 2:num_coeffs
            gpu_p1, gpu_p0 = gpuforwardrecurrence_next(n, A, B, C, gpu_z, gpu_p0, gpu_p1, num_points), gpu_p1
            gpu_f_z += (gpu_p1 .* c[n + 1])
        end
    end

    copyto!(f_z, gpu_f_z)
end

function forward_inplace(c::AbstractVector, A, B, C, z::Number, p0::Number, p1::Number)
    num_coeffs = length(c)
    
    f_z = zero(eltype(z))

    @inbounds for i in 2:num_coeffs-1
        p1, p0 = muladd(muladd(A[i], z, B[i]), p1, -C[i]*p0), p1
        f_z += p1*c[i+1]
    end

    return f_z
end