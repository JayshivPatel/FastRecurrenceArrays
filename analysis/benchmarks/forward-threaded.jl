using FastRecurrenceArrays, BenchmarkTools

# choose points
x = range(Float32(-10), Float32(10), 1000);

# num vectors
M = length(x);

# num recurrences
N = 100000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# parameters
params = (x, rec_U, N);

suite = BenchmarkGroup();

suite["row"] = @benchmarkable ThreadedRecurrenceArray(params..., 1) samples = 100 seconds = 500;
suite["column"] = @benchmarkable ThreadedRecurrenceArray(params..., 2) samples = 100 seconds = 500;

results = run(suite, verbose = true);

println("ThreadedRecurrenceArrayRow - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["row"]);

println("ThreadedRecurrenceArrayColumn - " * string(N) * " recurrences on " * string(M) * " points.");
display(results["column"]);