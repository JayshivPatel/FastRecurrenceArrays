using ClassicalOrthogonalPolynomials, FastRecurrenceArrays, RecurrenceRelationships, RecurrenceRelationshipArrays, Test

# forward recurrence - no data
x = [0.1+0im, 1.0001, 10.0];
M = length(x);
N = 15;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

@test FixedRecurrenceArray(x[1], rec_U, N) == chebyshevu.(0:N-1, x[1])

# forward recurrence - data (only check first few to avoid backwards swap)
ξ = @. inv(x + sign(x)sqrt(x^2-1));
@test FixedRecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2]).data[1:10, :] ≈ RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6

# GPU forward recurrence correctness
# TODO

# clenshaw
@test FixedClenshaw(Float32.(inv.(1:N)), rec_U, x).data ≈ clenshaw(Float32.(inv.(1:N)), rec_U..., x) 

# GPU clenshaw correctness
# TODO

# forward-inplace correctness
@test FixedClenshaw(Float32.(inv.(1:N)), rec_U, x).data ≈ ForwardInplace(Float32.(inv.(1:N)), rec_U, x).f_z