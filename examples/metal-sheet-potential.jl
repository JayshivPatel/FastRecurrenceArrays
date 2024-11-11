using SingularIntegrals, ClassicalOrthogonalPolynomials, InfiniteArrays, Test, Plots
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
f_ϵ = expand(T, Greens);

# Expand Log_Transform(1 - t^2)^(-1/2) in t
L_T = expand(T, I);

# Calculate u
# u = L_T^-1 (f_ϵ + k * e_0)
f_ϵ.args[2][1] += k*ϵ;
u = L_T \ f_ϵ;

# Evaluate v(x,y) pointwise using the Cauchy Transform of u
