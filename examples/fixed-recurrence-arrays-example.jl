using FixedRecurrenceArrays, RecurrenceRelationshipArrays, Test;

z = [0.1+0im, 1.0001, 10.0];

# Recurrence coefficients for ChebyshevU
rec_P = 2 * ones(100), zeros(100), ones(100);

# Exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[2] + sign(z[2]) * sqrt(z[2]^2-1));
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2-1));


@time x_vec = FixedRecurrenceArray(z[2], rec_P, [stieltjes, stieltjes^2], 10);
@time x_matrix = FixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2], 10);

@test x_vec ≈ RecurrenceArray(z[2], rec_P, [stieltjes, stieltjes^2])[1:10];
@test x_matrix ≈ RecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2])[1:10, :];