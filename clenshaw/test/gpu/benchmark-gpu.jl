using FixedClenshawArrays, BenchmarkTools, BenchmarkPlots

# choose points
x = Float32.(collect(-50.0:0.00005:50.0));

# num vectors
M = length(x);

# num recurrences
N = 1000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# gpu
gpu = @benchmarkable GPUFixedClenshaw(Float32.(inv.(1:N)), rec_U..., x) samples = 1000 evals = 1 seconds = 600;

gpu_display = run(gpu);
println("GPUFixedClenshaw - " * string(N) * " recurrences on " * string(M) * " points.");
display(gpu_display);