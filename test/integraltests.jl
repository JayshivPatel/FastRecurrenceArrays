# Basic unit tests

using ClassicalOrthogonalPolynomials, 
    FastRecurrenceArrays,
    RecurrenceRelationships,
    RecurrenceRelationshipArrays,
    SingularIntegrals,
    Test;

x = [0.1+0im, 1.0001, 10.0];
N = 15;
P = Legendre(); x = axes(P, 1); f = expand(P, exp); ff = collect(f.args[2][1:N-2]);

@testset "Stieltjes" begin
    @test inv.(10 .- x') * f ≈ FixedStieltjes(N, [10.0], ff).f[1]
end

@testset "LogKernel" begin
    @test log.(abs.(10 .- x')) * f ≈ InplaceLogKernel(N, [10.0], ff).f[1]
end
