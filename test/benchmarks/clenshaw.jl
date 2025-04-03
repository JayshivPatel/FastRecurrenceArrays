using FastRecurrenceArrays, BenchmarkTools

# choose points
x = Float32.(-10.0:0.005:10.0);

# num vectors
M = length(x);

# num recurrences
N = 1000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# parameters
params = (Float32.(inv.(1:N)), rec_U, x);

suite = BenchmarkGroup();

suite["fixed"] = @benchmarkable FixedClenshaw(params...) samples = 100 seconds = 500;
suite["gpu"] = @benchmarkable GPUClenshaw(params...) samples = 100 seconds = 500;

results = run(suite, verbose = true);

println("FixedClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["fixed"]);

println("GPUClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["gpu"]);
