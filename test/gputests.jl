# GPU unit tests

import ClassicalOrthogonalPolynomials: chebyshevu, expand, Legendre
import CUDA: has_cuda, has_cuda_gpu
import FastRecurrenceArrays: GPURecurrenceArray, GPUClenshaw, GPUInplaceStieltjes, GPUInplaceLogKernel
import RecurrenceRelationships: clenshaw
import RecurrenceRelationshipArrays: RecurrenceArray
import Test: @test, @testset

@assert has_cuda()
@assert has_cuda_gpu()

@testset "GPU" begin
    x = Float32.([1.0001, 5.0, 10.0]);
    M = length(x);
    N = 15;
    rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));
    @testset "Forward" begin
        # GPU forward recurrence - no data
        @test GPURecurrenceArray([x[1]], rec_U, N) ≈ chebyshevu.(0:N-1, x[1]);

        # GPU forward recurrence - data
        # only check first few to avoid backwards swap in adaptive version
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test GPURecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2]).data[1:10, :] ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1;
    end

    @testset "Clenshaw" begin
        # GPU clenshaw - no data
        @test GPUClenshaw(Float32.(inv.(1:N)), rec_U..., x).f ≈ clenshaw(Float32.(inv.(1:N)), rec_U..., x)

        # GPU clenshaw - data
        @test GPUClenshaw(Float32.(inv.(1:N)), rec_U..., [x[1]], [x[2]], [x[3]]).f[1] ≈ 
        (collect(inv.(1:N))' * RecurrenceArray(x[1], rec_U, x[2:3])[1:N]);
    end

    P = Legendre(); f = expand(P, exp); ff = Float32.(collect(f.args[2][1:N-2]));

    @testset "Integrals" begin
        @testset "Stieltjes" begin
            # GPU forward inplace (stieltjes)
            @test inv.(10 .- axes(P, 1)') * f ≈ GPUInplaceStieltjes(N, [Float32(10.0)], ff)[1]
        end
        @testset "LogKernel" begin
            # GPU forward inplace (logkernel)
            @test log.(abs.(10 .- axes(P, 1)')) * f ≈ GPUInplaceLogKernel(N, [Float32(10.0)], ff).f[1]
        end
    end
end