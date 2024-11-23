include("FixedRecurrenceArrays.jl");


z = (1:100);

# Recurrence coefficients for Legendre
rec_P = ones(10), (1:2:10), -1 * ones(10);

# Exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[1] + sign(z[1]) * sqrt(z[1]^2-1));
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2-1));


x_matrix = FixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2])
x_vec = FixedRecurrenceArray(z[1], rec_P, [stieltjes, stieltjes^2])