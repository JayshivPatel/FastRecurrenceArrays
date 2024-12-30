using BenchmarkTools, BenchmarkPlots, 
    StatsPlots, InfiniteArrays, Distributed, Test

# add remote processes created with Docker
addprocs(2);

# activate the M4R environment 
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());

# load modules
@everywhere using FixedRecurrenceArrays;

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
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2 - 1));

# distributed
matrix_result = @benchmarkable PartitionedFixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix' .^ 2], N) samples = 20 evals = 1 seconds = 60;

matrix_display = run(matrix_result);
println("PartitionedFixedRecurrenceArray - " * string(N) * "×" * string(M));
display(matrix_display);

rmprocs(2, 3);