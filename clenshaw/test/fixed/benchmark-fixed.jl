using FixedClenshawArrays, BenchmarkTools, BenchmarkPlots

# choose points
x = collect(-50.0:0.00005:50.0);

# num vectors
M = length(x);

# num recurrences
N = 1000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(N), zeros(N), ones(N+1));

# fixed
fixed = @benchmarkable FixedClenshaw(inv.(1:N), rec_U..., x) samples = 1000 evals = 1 seconds = 600;

fixed_display = run(fixed);
println("FixedClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
display(fixed_display);