using BenchmarkTools, 
    ClassicalOrthogonalPolynomials, 
    FastRecurrenceArrays,
    Test;

# choose points
x = ComplexF32.((10.0+0.00001:0.00001:12.0) .+ 0im);

# num vectors
M = length(x);

# Weighted OP basis, choose f(x) = exp(x)
P = Legendre(); f = expand(P, exp);

suite = BenchmarkGroup();

suite["default"] = @benchmarkable [real(inv.(x₀ .- axes(P, 1)') * f) for x₀ in x] samples = 100 seconds = 500;

# integral domain:
δx = Float32(2e-3);
xs = Float32.(-1.0:δx:1.0-δx);

suite["riemann"] = @benchmarkable [real(sum(@. inv(x₀ - xs) * exp(xs))) * δx for x₀ in x] samples = 100 seconds = 500;

stieltjesresults = run(suite, verbose = true);

println("Default - on " * string(M) * " points.");
display(stieltjesresults["default"]);

println("Riemann - with: " * string(length(xs)) * "estimates on " * string(M) * " points.");
display(stieltjesresults["riemann"]);

suite = BenchmarkGroup();

suite["default"] = @benchmarkable [real(log.(abs.(x₀ .- axes(P, 1)')) * f) for x₀ in x] samples = 100 seconds = 500;

suite["riemann"] = @benchmarkable [real(sum(@. log(abs(x₀ - xs)) * exp(xs))) * δx for x₀ in x] samples = 100 seconds = 500;

logkernelresults = run(suite, verbose = true);

println("Default - on " * string(M) * " points.");
display(logkernelresults["default"]);

println("Riemann - with: " * string(length(xs)) * "estimates on " * string(M) * " points.");
display(logkernelresults["riemann"]);