using FastRecurrenceArrays, BenchmarkTools

# choose points
x = Float32.(-10.0:0.02:(10.0-0.02));

# num vectors
M = length(x);

# num recurrences
N = 1000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# parameters
params = (x, rec_U, N);

suite = BenchmarkGroup();

suite["fixed"] = @benchmarkable FixedRecurrenceArray(params...) samples = 100 seconds = 500;
suite["gpu"] = @benchmarkable GPURecurrenceArray(params...) samples = 100 seconds = 500;

results = run(suite, verbose = true);

println("FixedRecurrenceArray - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["fixed"]);

println("GPURecurrenceArray - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["gpu"]);
