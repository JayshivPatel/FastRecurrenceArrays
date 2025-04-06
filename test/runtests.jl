# Basic unit tests

using ClassicalOrthogonalPolynomials, FastRecurrenceArrays, RecurrenceRelationships, RecurrenceRelationshipArrays, Test

x = [0.1+0im, 1.0001, 10.0];
M = length(x);
N = 15;
rec_U = (2 * ones(N), zeros(N), ones(N+1));

@testset "Forward" begin
    # forward recurrence - no data
    @test FixedRecurrenceArray(x[1], rec_U, N) == chebyshevu.(0:N-1, x[1]);

    # forward recurrence - data (only check first few to avoid backwards swap in adaptive version)
    ξ = @. inv(x + sign(x)sqrt(x^2-1));
    @test FixedRecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2]).data[1:10, :] ≈ 
        RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;
end

@testset "Clenshaw" begin
    # clenshaw
    @test FixedClenshaw(inv.(1:N), rec_U, x).data ≈ clenshaw(inv.(1:N), rec_U..., x);

    # forward-inplace correctness
    @test FixedClenshaw(inv.(1:N), rec_U, x).data ≈ ForwardInplace(inv.(1:N), rec_U, x).f_z;
end
