using FastRecurrenceArrays, BenchmarkTools;

x = range(Float32(-10), Float32(10), 1_000_000);
M = length(x);
N = 10_000;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

params = (Float32.(inv.(1:N)), rec_U, x);

println("ForwardInplace - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(@benchmark ForwardInplace($params...) samples = 100 seconds = 500));

println("ThreadedInplace - " * string(N) * " recurrences on " * string(M) * " points.");
println("Threads - " * string(Threads.nthreads()));
println(median(@benchmark ThreadedInplace($params...) samples = 100 seconds = 500));

println("GPUInplace - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(@benchmark GPUInplace($params...) samples = 100 seconds = 500));