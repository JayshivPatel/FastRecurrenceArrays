# Basic unit tests

import ClassicalOrthogonalPolynomials: Legendre, expand
import FastRecurrenceArrays: FixedStieltjes, InplaceStieltjes, FixedLogKernel, InplaceLogKernel
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@testset "Integrals" begin
    x = [0.1+0im, 1.0001, 10.0];
    N = 15;
    P = Legendre(); x = axes(P, 1); f = expand(P, exp); ff = collect(f.args[2][1:N-2]);

    @testset "Stieltjes" begin
        # Forward recurrence (stieltjes)
        @test inv.(10 .- x') * f ≈ FixedStieltjes(N, [10.0], ff)[1];
        # Forward inplace (stieltjes)
        @test FixedStieltjes(N, [10.0], ff)[1] ≈ InplaceStieltjes(N, [10.0], ff)[1];
    end

    @testset "LogKernel" begin
        # Forward recurrence (logkernel)
        @test log.(abs.(10 .- x')) * f ≈ FixedLogKernel(N, [10.0], ff)[1];
        # Forward inplace (logkernel)
        @test FixedLogKernel(N, [10.0], ff)[1] ≈ InplaceLogKernel(N, [10.0], ff)[1];
    end
end