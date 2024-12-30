using FixedRecurrenceArrays, BenchmarkTools, BenchmarkPlots, 
    StatsPlots, InfiniteArrays

# choose points inside the domain: (make complex)
z_in = (-1.0:0.05:1.0) .+ 0 * im;

# choose points outside the domain:
z_out = (1.0:0.05:2.0);

z = [z_in; z_out];

# num vectors
M = length(z);

# num recurrences
N = 100000;

# recurrence coefficients for Legendre
rec_P = (1:10000), (1:2:20000), -1 * (1:10000);

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[1] + sign(z[1]) * sqrt(z[1]^2 - 1));
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2 - 1));

# fixed
vector_result = @benchmarkable FixedRecurrenceArray(z[1], rec_P, [stieltjes, stieltjes^ 2], N) samples = 10000 evals = 1 seconds = 60;
matrix_result = @benchmarkable FixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix' .^ 2], N) samples = 20 evals = 1 seconds = 60;

vector_display = run(vector_result);
println("FixedRecurrenceVector - " * string(N) * "×" * string(1));
display(vector_display);

matrix_display = run(matrix_result);
println("FixedRecurrenceMatrix - " * string(N) * "×" * string(M));
display(matrix_display);
