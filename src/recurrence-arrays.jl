using RecurrenceRelationshipArrays, FillArrays, InfiniteArrays

# Choose a point close to the support
z = 1.0001;

# Recurrence coefficients for Legendre
rec_P = (1:∞), (1:2:∞), -1 * (1:∞);

# Exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z + sign(z)sqrt(z^2 - 1));

@time r = RecurrenceArray(z, rec_P, [stieltjes, stieltjes^2]);

