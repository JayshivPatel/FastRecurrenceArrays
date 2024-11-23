include("FixedRecurrenceArrays.jl");


z = 2;

# Recurrence coefficients for Legendre
rec_P = [2], [0], [1];

# Exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z + sign(z) * sqrt(z^2-1));

x = FixedRecurrenceArray(z, rec_P, [stieltjes', stieltjes'.^2])