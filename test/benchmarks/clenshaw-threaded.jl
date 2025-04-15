using FastRecurrenceArrays, BenchmarkTools

# choose points
x = Float32.(-10.0:0.02:(10.0-0.02));

# num vectors
M = length(x);

# num recurrences
N = 100000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# parameters
params = (Float32.(inv.(1:N)), rec_U..., x);

suite = BenchmarkGroup();

suite["row"] = @benchmarkable ThreadedClenshaw(params..., 1) samples = 100 seconds = 500;
suite["column"] = @benchmarkable ThreadedClenshaw(params..., 2) samples = 100 seconds = 500;

results = run(suite, verbose = true);

println("ThreadedClenshawRow - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["row"]);

println("ThreadedClenshawColumn - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["column"]);
