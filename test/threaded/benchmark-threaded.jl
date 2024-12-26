using FixedRecurrenceArrays, BenchmarkTools, BenchmarkPlots, 
    StatsPlots, InfiniteArrays

# choose points inside the domain: (make complex)
z_in = (-1.0:0.0005:1.0) .+ 0 * im;

# choose points outside the domain:
z_out = (1.0:0.0005:2.0);

z = [z_in; z_out];

# num vectors
M = length(z)

# num recurrences
N = 10000

# recurrence coefficients for Legendre
rec_P = (1:10000), (1:2:20000), -1 * (1:10000);

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2 - 1));

# threaded
result = @benchmarkable ThreadedFixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix' .^ 2], N) samples = 20 evals = 1 seconds = 60;

disp = run(result)

println("ThreadedFixedRecurrenceArray - " * string(N) * "×" * string(M));
display(disp);