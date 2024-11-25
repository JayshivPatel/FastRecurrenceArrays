using FixedRecurrenceArrays;

z = [0.1+0im, 1.0001, 10.0];

# Recurrence coefficients for ChebyshevU
rec_P = 2 * ones(100), zeros(100), ones(100);

# Exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[1] + sign(z[1]) * sqrt(z[1]^2-1));
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2-1));


@time x_vec = FixedRecurrenceArray(z[1], rec_P, [stieltjes, stieltjes^2], 100);
@time x_matrix = FixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2], 100);
