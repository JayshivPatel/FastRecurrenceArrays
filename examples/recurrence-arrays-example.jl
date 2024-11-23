using RecurrenceRelationshipArrays, FillArrays, InfiniteArrays, Test

# Choose points inside the domain: (make complex)
z_in = (-1.0005:0.01:1.0005) .+ 0*im;

# Choose points outside the domain:
z_out = (1.0:0.01:100.0);

z = [z_in; z_out];

# Recurrence coefficients for Legendre
rec_P = (1:∞), (1:2:∞), -1 * (1:∞);

# Exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = @. inv(z + sign(z)sqrt(z^2-1));

# Evaluate multiple solutions sequentially:
@time [RecurrenceArray(z[i], rec_P, [stieltjes'[i], stieltjes'[i]^2])[1:100] for i in eachindex(z)];

# Evaluate multiple solutions concurrently.
@time r = RecurrenceArray(z, rec_P, [stieltjes'; stieltjes'.^2]);

# Assert correctness when compared to serial calculation:
for i in eachindex(z)
    @test isequal(r[1:10, i], RecurrenceArray(z[i], rec_P, [stieltjes'[i], stieltjes'[i]^2])[1:10]);
end

