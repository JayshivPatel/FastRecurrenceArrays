# GPU unit tests

using CUDA, ClassicalOrthogonalPolynomials, FastRecurrenceArrays, RecurrenceRelationships, RecurrenceRelationshipArrays, Test

x = [0.1+0im, 1.0001, 10.0];
M = length(x);
N = 15;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

@assert CUDA.has_cuda()
@assert CUDA.has_cuda_gpu()

@testset "Forward" begin
    # GPU forward recurrence - no data
    @test GPURecurrenceArray(x[1], rec_U, N) == chebyshevu.(0:N-1, x[1]);

    # GPU forward recurrence - data (only check first few to avoid backwards swap in adaptive version)
    ξ = @. inv(x + sign(x)sqrt(x^2-1));
    @test GPURecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2]).data[1:10, :] ≈ 
        RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;
end

@testset "Clenshaw" begin
    # GPU clenshaw correctness
    @test GPUClenshaw(Float32.(inv.(1:N)), rec_U, x).data ≈ clenshaw(Float32.(inv.(1:N)), rec_U..., x)
end
