using FastRecurrenceArrays, BenchmarkTools

# choose points
x = range(Float32(-10), Float32(10), 10);

# num vectors
M = length(x);

# num recurrences
N = 10;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# parameters
params = (x, rec_U, N);

println("FixedRecurrenceArray - " * string(N) * " recurrences on " * string(M) * " points.");
display(@benchmark FixedRecurrenceArray(params...) samples = 100 seconds = 500);

println("GPURecurrenceArray - " * string(N) * " recurrences on " * string(M) * " points.");
display(@benchmark GPURecurrenceArray(params...) samples = 100 seconds = 500);