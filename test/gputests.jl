# GPU unit tests

import ClassicalOrthogonalPolynomials: chebyshevu, expand, Legendre
import CUDA: has_cuda, has_cuda_gpu
import FastRecurrenceArrays: GPURecurrenceArray, GPUClenshaw, GPUInplace, GPUInplaceCauchy, GPUInplaceLogKernel
import LinearAlgebra: dot
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@assert has_cuda()
@assert has_cuda_gpu()

@testset "GPU" begin
    x = ComplexF32.([0.1+0.1im, 1.0001, 10.0]);
    N = 15;
    rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));
    @testset "Forward" begin
        # GPU forward recurrence - no data
        @test Array(GPURecurrenceArray([x[1]], rec_U, N)) ≈ chebyshevu.(0:N-1, x[1]);

        # GPU forward recurrence - data
        # only check first few to avoid backwards swap in adaptive version
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test Array(GPURecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2]))[1:4, :] ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:4, :] atol=1e-6;
    end

    @testset "Clenshaw" begin
        # GPU clenshaw
        @test Array(GPUClenshaw(Float32.(inv.(1:N)), rec_U, x)) ≈ clenshaw(Float32.(inv.(1:N)), rec_U..., x)
    end

    @testset "Inplace" begin   
        # forward-inplace - no data
        @test Array(GPUInplace(Float32.(inv.(1:N)), rec_U, x)) ≈ clenshaw(inv.(1:N), rec_U..., x);

        # forward-inplace - data
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test Array(GPUInplace(Float32.(inv.(1:N)), rec_U, x, [ξ'; ξ'.^2]))[1] ≈ 
            dot(inv.(1:N), RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:N, 1]);
    end

    P = Legendre(); f = expand(P, exp); ff = Float32.(collect(f.args[2][1:N-2]));

    @testset "Integrals" begin
        @testset "Cauchy" begin
            # GPU forward inplace (cauchy)
            @test -inv(2π*im) * (inv.(x[1] .- axes(P, 1)') * f) ≈ Array(GPUInplaceCauchy(N, [x[1]], ff))[1]
        end
        @testset "LogKernel" begin
            # GPU forward inplace (logkernel)
            @test log.(abs.(x[1] .- axes(P, 1)')) * f ≈ Array(GPUInplaceLogKernel(N, [x[1]], ff))[1]
        end
    end
end