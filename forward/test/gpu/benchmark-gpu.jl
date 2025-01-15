using FixedRecurrenceArrays, BenchmarkTools, BenchmarkPlots

# choose points inside the domain: (make complex)
z_in = ComplexF32.(collect((-1.0:0.005:1.0) .+ 0 * im));

# choose points outside the domain:
z_out = Float32.(10.0:0.005:50.0);

z = [z_in; z_out];

# num vectors
M = length(z);

# num recurrences
N = 10000;

# recurrence coefficients for Legendre
rec_P = (1:10000), (1:2:20000), -1 * (1:10000);

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[1] + sign(z[1]) * sqrt(z[1]^2 - 1));
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2 - 1));

# threaded
result = @benchmarkable GPUFixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix' .^ 2], N) samples = 1000 evals = 1 seconds = 600;

disp = run(result);

println("GPUFixedRecurrenceArray - " * string(N) * "×" * string(M));
display(disp);