# Partitioned unit tests

using Distributed, ClassicalOrthogonalPolynomials, FastRecurrenceArrays, RecurrenceRelationships,
    RecurrenceRelationshipArrays, Test

@testset "Partitioned" begin
    x = [0.1+0im, 1.0001, 10.0];
    M = length(x);
    N = 15;
    rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

    # add a remote process created with Docker
    addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2222", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
    addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2223", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");

    # activate the FastRecurrenceArrays environment 
    @everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
    # load module
    @everywhere using FastRecurrenceArrays;

    @assert nprocs() > 1
    @testset "Forward" begin
        # Partitioned forward recurrence - no data
        @test vec(PartitionedRecurrenceArray(x, rec_U, N)[:, 1]) == chebyshevu.(0:N-1, x[1]);

        # Partitioned forward recurrence - data (only check first few to avoid backwards swap in adaptive version)
        ξ = @. inv(x + sign(x)sqrt(x^2-1));
        @test PartitionedRecurrenceArray(x, rec_U, N, [ξ'; ξ'.^2])[1:10, :] ≈ 
            RecurrenceArray(x, rec_U, [ξ'; ξ'.^2])[1:10, :] atol=1e-6;
    end
end