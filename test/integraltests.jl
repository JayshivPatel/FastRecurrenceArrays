# Basic unit tests

import ClassicalOrthogonalPolynomials: Legendre, expand
import FastRecurrenceArrays: FixedStieltjes, InplaceStieltjes, FixedLogKernel, InplaceLogKernel
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@testset "Integrals" begin
    x = [0.1+0.1im, 1.0001, 10.0];
    N = 15;
    P = Legendre(); f = expand(P, exp); ff = collect(f.args[2][1:N-2]);

    @testset "Stieltjes" begin
        # Forward recurrence (stieltjes)
        @test inv.(x[1] .- axes(P, 1)') * f ≈ FixedStieltjes(N, [x[1]], ff);
        # Forward inplace (stieltjes)
        @test inv.(x[1] .- axes(P, 1)') * f ≈ InplaceStieltjes(N, [x[1]], ff)[1];
    end

    @testset "LogKernel" begin
        # Forward recurrence (logkernel)
        @test log.(abs.(x[1] .- axes(P, 1)')) * f ≈ FixedLogKernel(N, [x[1]], ff);
        # Forward inplace (logkernel)
        @test log.(abs.(x[1] .- axes(P, 1)')) * f ≈ InplaceLogKernel(N, [x[1]], ff)[1];
    end
end