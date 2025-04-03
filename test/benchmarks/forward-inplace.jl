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
params = (Float32.(inv.(1:N)), rec_U, x);

result = @benchmarkable ForwardInplace(params...) samples = 100 seconds = 500;

result_display = run(result, verbose=true)
println("ForwardInplace - " * string(N) * "×" * string(M))
display(result_display);