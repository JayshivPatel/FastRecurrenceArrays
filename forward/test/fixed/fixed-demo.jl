using FixedRecurrenceArrays

# choose points inside the domain: (make complex)
z_in = (-5.0005:0.001:5.0005) .+ 0 * im

# choose points outside the domain:
z_out = (1.0:0.001:100.0)

z = [z_in; z_out]

# recurrence coefficients for Legendre
rec_P = (1:10000), (1:2:10000), -1 * (1:10000)

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[1] + sign(z[1]) * sqrt(z[1]^2 - 1))
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2 - 1))

FixedRecurrenceArray(z[1], rec_P, [stieltjes, stieltjes^2], 1000)
FixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix' .^ 2], 1000)