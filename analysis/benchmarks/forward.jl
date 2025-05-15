using FastRecurrenceArrays, BenchmarkTools;

x = range(Float32(-10), Float32(10), 1_000_000);
M = length(x);
N = 10;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

params = (x, rec_U, N);

println("FixedRecurrenceArray - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(run(@benchmarkable FixedRecurrenceArray($params...) samples = 100)));

println("ThreadedRecurrenceArray (row-wise) - " * string(N) * " recurrences on " * string(M) * " points.");
println("Threads - " * string(Threads.nthreads()));
println(median(run(@benchmarkable ThreadedRecurrenceArray($params..., Val(1)) samples = 100)));

println("ThreadedRecurrenceArray (column-wise) - " * string(N) * " recurrences on " * string(M) * " points.");
println("Threads - " * string(Threads.nthreads()));
println(median(run(@benchmarkable ThreadedRecurrenceArray($params..., Val(2)) samples = 100)));

println("GPURecurrenceArray - " * string(N) * " recurrences on " * string(M) * " points.");
println(median(run(@benchmarkable GPURecurrenceArray($params...) samples = 100)));