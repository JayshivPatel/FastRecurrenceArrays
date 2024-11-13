using SingularIntegrals, ClassicalOrthogonalPolynomials, ContinuumArrays, InfiniteArrays, Test, Plots
# z = x + iy
# v_xx + v_yy = 0 for z not in [-1,1] OR {ie}
# v(x,y) = log|z-ie| + O(1) as z -> ie
# v(x,y) = o(1) as z -> inf
# v(x,0) = k for -1 < x < 1, k an unknown constant.

# Define a small value for ϵ.
ϵ = 0.1;

# Define k.
k = 5;

# Define the Green's function.
function Greens(x)
    return - log(abs(x-ϵ*im));
end

# Define the integral operator for T_n(x)
function I(x)
    return (1 + x^2)^(-1/2);
end

# Choose a random point z 
z = 0.4 + 0.2*im;

# Use the ChebyshevT OPs
T = ChebyshevT(); 

# Expand f_ϵ in T
f_ϵ = expand(T, Greens)

# Expand Log_Transform(1 - t^2)^(-1/2) in t
L_T = expand(T, I)

# Calculate u
# u = L_T^-1 (f_ϵ + k * e_0)
coefficients(f_ϵ)[1] += k*ϵ;
u = L_T \ f_ϵ;

# Evaluate v(x,y) pointwise using the Cauchy Transform of u



# Kernel:

t = axes(T,1)
L = log.(abs.(t .- t'))  # L[x,y] == log(abs(x-y))
@test L[0.1,0.2] == log(abs(0.1-0.2))

W = Weighted(T) # T multiplied by the orthogonality weight 1/sqrt(1-x^2)
@test W[0.1,1:5] == 1/sqrt(1-0.1^2) * T[0.1,1:5]
T \ L * W  # gives matrix
T \ logkernel(W) # same as above




# pontwise-version

P = Legendre()
f_ϵ = expand(P, Greens)
L_ϵ = log.(3.0 .- t')
L_ϵ*f_ϵ

@test logkernel(f_ϵ, im*ϵ) == logkernel(P, im*ϵ) * coefficients(f_ϵ)

@time for k = 1:100^2
    transpose(complexlogkernel(P, randn() + im*randn()))[1:1000]
end


