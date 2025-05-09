# Threaded unit tests

import ClassicalOrthogonalPolynomials: chebyshevu
import FastRecurrenceArrays: ThreadedRecurrenceArray, ThreadedClenshaw, ThreadedInplace
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@assert Threads.nthreads() > 1

@testset "ThreadedTests" begin
    x = [0.1+0im, 1.0001, 10.0];
    M = length(x);
    N = 15;
    rec_U = (2 * ones(N), zeros(N), ones(N+1));

    @testset "Forward" begin
        # forward recurrence - no data (rowwise)
        @test permutedims(ThreadedRecurrenceArray([x[1]], rec_U, N, 1)) ≈ chebyshevu.(0:N-1, x[1]);

        # forward recurrence - no data (columnwise)
        @test ThreadedRecurrenceArray([x[1]], rec_U, N, 2) ≈ chebyshevu.(0:N-1, x[1]);

        # forward recurrence - data (rowwise; only check first few to avoid backwards swap in adaptive version)
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test permutedims(ThreadedRecurrenceArray(x, rec_U, N, 1, [ξ'; ξ'.^2]).data[:, 1:10]) ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;

        # forward recurrence - data (columnwise)
        @test ThreadedRecurrenceArray(x, rec_U, N, 2, [ξ'; ξ'.^2]).data[1:10, :] ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;
    end

    @testset "Clenshaw" begin
        # clenshaw - no data (columnwise)
        @test ThreadedClenshaw(inv.(1:N), rec_U..., x).f ≈ clenshaw(inv.(1:N), rec_U..., x);

        # clenshaw - data (columnwise)
        @test ThreadedClenshaw(inv.(1:N), rec_U..., [x[1]], [x[2]], [x[3]]).f[1] ≈ 
            (collect(inv.(1:N))' * RecurrenceArray(x[1], rec_U, x[2:3])[1:N]);

        # forward-inplace correctness (columnwise)
        @test ThreadedClenshaw(inv.(1:N), rec_U..., x).f ≈ ThreadedInplace(inv.(1:N), rec_U, x).f;
    end
end