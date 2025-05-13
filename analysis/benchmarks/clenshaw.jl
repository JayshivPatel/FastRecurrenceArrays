using FastRecurrenceArrays, BenchmarkTools;

x = range(Float32(-10), Float32(10), 1_000_000);
M = length(x);
N = 10;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

params = (Float32.(inv.(1:N)), rec_U, x);

println("FixedClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(@benchmark FixedClenshaw($params[1], $params[2], $(collect(params[3]))) samples = 100 seconds = 100));

println("ThreadedClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
println("Threads - " * string(Threads.nthreads()));
println(median(@benchmark ThreadedClenshaw($params...) samples = 100 seconds = 100));

println("GPUClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(@benchmark GPUClenshaw($params...) samples = 100 seconds = 100));