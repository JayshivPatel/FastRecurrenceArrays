using FastRecurrenceArrays, BenchmarkTools;

x = range(Float32(-10), Float32(10), 1_000_000);
M = length(x);
N = 10;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

params = (Float32.(inv.(1:N)), rec_U, x);

println("FixedClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(run(@benchmarkable FixedClenshaw($params[1], $params[2], $(collect(params[3]))) samples = 100 seconds = 500)));

println("ThreadedClenshaw (row-wise) - " * string(N) * " recurrences on " * string(M) * " points.");
println("Threads - " * string(Threads.nthreads()));
println(median(run(@benchmarkable ThreadedClenshaw($params..., Val(1)) samples = 100 seconds = 500)));

println("ThreadedClenshaw (column-wise) - " * string(N) * " recurrences on " * string(M) * " points.");
println("Threads - " * string(Threads.nthreads()));
println(median(run(@benchmarkable ThreadedClenshaw($params..., Val(2)) samples = 100 seconds = 500)));

println("GPUClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(run(@benchmarkable GPUClenshaw($params...) samples = 100 seconds = 500)));