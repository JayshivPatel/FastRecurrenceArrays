# FastRecurrenceArrays
A Julia package for performantly computing recurrence relationships and singular integrals.

This package supports creating recurrence arrays, function approximation and computing singular integrals involving Cauchy and log transforms.

Note: `GPURecurrenceArray` and other GPU-based computations will operate on 32-bit floating-point numbers *only*.

Note: the following commands were pre-compiled.

## Recurrence Arrays
For example, we compute the first 15 Chebyshev U polynomials at one million points within the integral domain.

```julia
julia> using FastRecurrenceArrays

julia> N = 15; rec_U = (2 * ones(N), zeros(N), ones(N+1));

julia> x = range(-1, 1, 1_000_000);

julia> @time FixedRecurrenceArray(x, rec_U, N)
  0.210338 seconds (6 allocations: 122.070 MiB, 3.07% gc time)
15×1000000 Matrix{Float64}:
   1.0    1.0        1.0        1.0        1.0      …   1.0       1.0       1.0       1.0       1.0       1.0
  -2.0   -2.0       -1.99999   -1.99999   -1.99998      1.99998   1.99998   1.99999   1.99999   2.0       2.0
   3.0    2.99998    2.99997    2.99995    2.99994      2.99992   2.99994   2.99995   2.99997   2.99998   3.0
    ⋮                                        ⋮            ⋮                                                  ⋮
  13.0   12.9985    12.9971    12.9956    12.9942      12.9927   12.9942   12.9956   12.9971   12.9985   13.0
 -14.0  -13.9982   -13.9964   -13.9945   -13.9927      13.9909   13.9927   13.9945   13.9964   13.9982   14.0
  15.0   14.9978    14.9955    14.9933    14.991    …  14.9888   14.991    14.9933   14.9955   14.9978   15.0

julia> @time ThreadedRecurrenceArray(x, rec_U, N, Val(1));
  0.065084 seconds (553 allocations: 122.131 MiB, 23.90% gc time)
julia> @time ThreadedRecurrenceArray(x, rec_U, N, Val(2));
  0.128255 seconds (48 allocations: 122.075 MiB, 12.84% gc time)
julia> @time GPURecurrenceArray(x, rec_U, N);
┌ Warning: Converting input vector(s) to Float32 for improved performance...
│   x = -1.0:2.000002000002e-6:1.0
└ @ FastRecurrenceArrays ~/.julia/packages/FastRecurrenceArrays/y4Dpk/src/forward.jl:137
  0.004941 seconds (3.46 k allocations: 15.333 MiB)

# Additionally, a cluster of computers running Julia can be used to compute these in a distributed manner.
# As an example, we use Docker containers on the localhost, connecting to them via ssh.

julia> using Distributed

julia> addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2222", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
julia> addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2223", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");

julia> workers()
2-element Vector{Int64}:
 2
 3

julia> @everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
  Activating
  From worker 2: Activating project at `/tmp/FastRecurrenceArrays`
  From worker 3: Activating project at `/tmp/FastRecurrenceArrays`
  No Changes to `~/Project.toml`
  No Changes to `~/Manifest.toml`
julia> @everywhere using FastRecurrenceArrays;

julia> @time PartitionedRecurrenceArray(x, rec_U, N)
  0.099244 seconds (641 allocations: 15.299 MiB, 2.13% gc time, 4 lock conflicts)
15×1000000 PartitionedArray: no preview available on partitioned array.

```

## Function Approximation
Here we approximate the $\exp$ function using the Chebyshev U polynomials at one million points. This can be done using Clenshaw's algorithm and the forward recurrence, either with the dot product or in-place.

```julia
julia> using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, Test

julia> N = 15; rec_U = (2 * ones(N), zeros(N), ones(N+1));

julia> x = range(-1, 1, 1_000_000);

julia> U = ChebyshevU(); f_N = transform(U[:, 1:N], exp)
15-element Vector{Float64}:
 1.13031820798497
 0.5429906790681532
 0.13301054954599142
  ⋮
 5.188627305585669e-13
 1.9970136655445003e-14
 1.3877787807814457e-15

julia> @time FixedClenshaw(f_N, rec_U, x)
  0.024923 seconds (3 allocations: 7.629 MiB)
1000000-element Vector{Float64}:
 0.36787944117145077
 0.36788017693180464
 0.36788091269363
 ⋮
 2.718270955342615
 2.7182763918953987
 2.7182818284590557

julia> @time ThreadedClenshaw(f_N, rec_U, x, Val(1));
  0.064917 seconds (558 allocations: 30.581 MiB, 8.18% gc time)
julia> @time ThreadedClenshaw(f_N, rec_U, x, Val(2));
  0.010086 seconds (45 allocations: 7.634 MiB, 58.78% gc time)
julia> @time GPUClenshaw(f_N, rec_U, x);
  0.004185 seconds (4.37 k allocations: 3.920 MiB)
  
julia> @test FixedClenshaw(f_N, rec_U, x)[1:10] ≈ exp.(x[1:10])
Test Passed
julia> @test (transpose(FixedRecurrenceArray(x, rec_U, N)) * f_N)[1:10] ≈ exp.(x[1:10])
Test Passed

julia> @time ForwardInplace(f_N, rec_U, x);
  0.186778 seconds (12 allocations: 30.518 MiB, 81.16% gc time)
julia> @time ThreadedInplace(f_N, rec_U, x, Val(1));
  0.073874 seconds (561 allocations: 38.209 MiB, 14.07% gc time)
julia> @time ThreadedInplace(f_N, rec_U, x, Val(2));
  0.109188 seconds (54 allocations: 30.523 MiB, 77.96% gc time)
julia> @time GPUInplace(f_N, rec_U, x);
  0.005933 seconds (5.75 k allocations: 15.397 MiB)

julia> @test ForwardInplace(f_N, rec_U, x)[1:10] ≈ exp.(x[1:10])
Test Passed

```
## Cauchy and log Transforms
We approximate the Cauchy and log transforms of the $\exp$ function using the Legendre polynomials at one million points with 15 recurrences using the three different algorithms.

Note: Parallel implementations using multiple threads and on the GPU can be used but are not shown here.

Note: These implementations work exclusively with the Legendre polynomials.

```julia
julia> using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, Test

julia> N = 15; P = Legendre(); f_N = transform(P[:, 1:N], exp);

julia> ff = expand(P, exp); xs = axes(P, 1);

julia> x = range(-0.999 + 0im, 0.999 + 0im, 1_000_000);

julia> @time FixedCauchy(N, x, f_N)
  0.553041 seconds (27 allocations: 320.436 MiB, 9.19% gc time)
1000000-element Vector{ComplexF64}:
   4.154597849352282 + 1.15688365519706im
   4.153869142047141 + 1.1568859666552243im
   4.153141896016044 + 1.1568882781180059im
                     ⋮
 -17.044389072794655 + 8.531164666258856im
 -17.049838406487453 + 8.531181711559931im
 -17.055298559281507 + 8.531198756895066im
julia> @time ClenshawCauchy(N, x, f_N);
  0.288978 seconds (22 allocations: 91.554 MiB, 2.33% gc time)
julia> @time InplaceCauchy(N, x, f_N);
  0.295318 seconds (43 allocations: 198.366 MiB, 2.75% gc time)

# ∫₋₁¹ exp(t) / (t - x[1]) dt
julia> @test FixedCauchy(N, x, f_N)[1] ≈ -inv.(x[1] .- xs') * ff
Test Passed

julia> @time FixedLogKernel(N, x, f_N);
  0.675080 seconds (27 allocations: 328.065 MiB, 31.49% gc time)
julia> @time ClenshawLogKernel(N, x, f_N);
  0.345528 seconds (31 allocations: 114.443 MiB, 2.37% gc time)
julia> @time InplaceLogKernel(N, x, f_N);
  0.546135 seconds (49 allocations: 236.513 MiB, 33.79% gc time)

# ∫₋₁¹ exp(t) * log(t - x[1]) dt
julia> @test FixedLogKernel(N, x, f_N)[1] ≈ log.(abs.(x[1] .- xs')) * ff
Test Passed

```

