# Basic unit tests

import ClassicalOrthogonalPolynomials: Legendre, expand
import FastRecurrenceArrays: FixedCauchy, ClenshawCauchy, InplaceCauchy, FixedLogKernel, ClenshawLogKernel, InplaceLogKernel
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@testset "Integrals" begin
    x = [0.1+0.1im, 1.0001, 10.0];
    N = 15;
    P = Legendre(); f = expand(P, exp); ff = collect(f.args[2][1:N]);

    @testset "Cauchy" begin
        # Forward (cauchy)
        @test -(inv.(x[1] .- axes(P, 1)') * f) ≈ FixedCauchy(N, x, ff)[1];
        # Clenshaw (cauchy)
        @test -(inv.(x[1] .- axes(P, 1)') * f) ≈ ClenshawCauchy(N, x, ff)[1];
        # Forward inplace (cauchy)
        @test -(inv.(x[1] .- axes(P, 1)') * f) ≈ InplaceCauchy(N, x, ff)[1];
    end

    @testset "LogKernel" begin
        # Forward (logkernel)
        @test log.(abs.(x[1] .- axes(P, 1)')) * f ≈ FixedLogKernel(N, x, ff)[1];
        # Clenshaw (logkernel)
        @test log.(abs.(x[1] .- axes(P, 1)')) * f ≈ ClenshawLogKernel(N, x, ff)[1];
        # Forward inplace (logkernel)
        @test log.(abs.(x[1] .- axes(P, 1)')) * f ≈ InplaceLogKernel(N, x, ff)[1];
    end
end