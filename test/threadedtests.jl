# Threaded unit tests

import ClassicalOrthogonalPolynomials: chebyshevu
import FastRecurrenceArrays: ThreadedRecurrenceArray, ThreadedClenshaw, ThreadedInplace
import LinearAlgebra: dot
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@assert Threads.nthreads() > 1

@testset "ThreadedTests" begin
    x = [0.1+0.1im, 1.0001, 10.0];
    N = 15;
    rec_U = (2 * ones(N), zeros(N), ones(N+1));

    @testset "Forward" begin
        # forward recurrence - no data (rowwise)
        @test ThreadedRecurrenceArray([x[1]], rec_U, N, Val(1)) ≈ chebyshevu.(0:N-1, x[1]);

        # forward recurrence - no data (columnwise)
        @test ThreadedRecurrenceArray([x[1]], rec_U, N, Val(2)) ≈ chebyshevu.(0:N-1, x[1]);

        # forward recurrence - data (rowwise; only check first few to avoid backwards swap in adaptive version)
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test ThreadedRecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2], Val(1))[1:10, :] ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;

        # forward recurrence - data (columnwise)
        @test ThreadedRecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2], Val(2))[1:10, :] ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;
    end

    @testset "Clenshaw" begin
        # clenshaw (rowwise) - no data
        @test ThreadedClenshaw(inv.(1:N), rec_U, x, Val(1)) ≈ clenshaw(inv.(1:N), rec_U..., x);

        # clenshaw (columnwise) - no data
        @test ThreadedClenshaw(inv.(1:N), rec_U, x, Val(2)) ≈ clenshaw(inv.(1:N), rec_U..., x);

        # clenshaw (rowwise) - data
        @test ThreadedClenshaw(inv.(1:N), rec_U, [x[1]], [x[2]], [x[3]], Val(1))[1] ≈ 
            (collect(inv.(1:N))' * RecurrenceArray(x[1], rec_U, x[2:3])[1:N]);

        # clenshaw (columnwise) - data
        @test ThreadedClenshaw(inv.(1:N), rec_U, [x[1]], [x[2]], [x[3]], Val(2))[1] ≈ 
            (collect(inv.(1:N))' * RecurrenceArray(x[1], rec_U, x[2:3])[1:N]);
    end

    @testset "Inplace" begin   
        # forward-inplace (rowwise) - no data
        @test ThreadedInplace(inv.(1:N), rec_U, x, Val(1)) ≈ clenshaw(inv.(1:N), rec_U..., x);

        # forward-inplace (columnwise) - no data
        @test ThreadedInplace(inv.(1:N), rec_U, x,  Val(2)) ≈ clenshaw(inv.(1:N), rec_U..., x);

        # forward-inplace (rowwise) - data
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test ThreadedInplace(inv.(1:N), rec_U, x, [ξ'; ξ'.^2], Val(1))[1] ≈ 
            dot(inv.(1:N), RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:N, 1]);

        # forward-inplace (columnwise) - data
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test ThreadedInplace(inv.(1:N), rec_U, x, [ξ'; ξ'.^2], Val(2))[1] ≈ 
            dot(inv.(1:N), RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:N, 1]);
    end
end