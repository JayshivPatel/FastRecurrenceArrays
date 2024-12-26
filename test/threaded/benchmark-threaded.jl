using FixedRecurrenceArrays, BenchmarkTools, BenchmarkPlots, 
    StatsPlots, InfiniteArrays

# choose points inside the domain: (make complex)
z_in = (-5.0005:0.001:5.0005) .+ 0 * im;

# choose points outside the domain:
z_out = (1.0:0.001:100.0);

z = [z_in; z_out];

# num vectors
M = length(z)

# num recurrences
N = 1000

# recurrence coefficients for Legendre
rec_P = (1:1000), (1:2:2000), -1 * (1:1000);

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2 - 1));

# threaded
result = @benchmark ThreadedFixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix' .^ 2], N) samples = 100 evals = 1 seconds = 60;

println("ThreadedFixedRecurrenceArray - " * string(N) * "×" * string(M));
display(result);