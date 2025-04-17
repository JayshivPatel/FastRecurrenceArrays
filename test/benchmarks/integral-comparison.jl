using BenchmarkTools, 
    ClassicalOrthogonalPolynomials, 
    FastRecurrenceArrays,
    Test;

# choose points
x = ComplexF32.((10.0+0.00001:0.00001:12.0) .+ 0im);

# num vectors
M = length(x);

# num recurrences
N = 15;

# Weighted OP basis, choose f(x) = exp(x)
P = Legendre(); f = expand(P, exp); ff = Float32.(collect(f.args[2][1:N-2]));

suite = BenchmarkGroup();

suite["fixed"] = @benchmarkable FixedStieltjes(N, x, ff) samples = 100 seconds = 500;
suite["inplace"] = @benchmarkable InplaceStieltjes(N, x, ff) samples = 100 seconds = 500;
suite["threaded-inplace"] = @benchmarkable ThreadedInplaceStieltjes(N, x, ff) samples = 100 seconds = 500;
suite["GPU-inplace"] = @benchmarkable GPUInplaceStieltjes(N, x, ff) samples = 100 seconds = 500;

stieltjesresults = run(suite, verbose = true);

println("FixedStieltjes - " * string(N) * " recurrences on " * string(M) * " points.");
display(stieltjesresults["fixed"]);

println("InplaceStieltjes - " * string(N) * " recurrences on " * string(M) * " points.");
display(stieltjesresults["inplace"]);

println("ThreadedInplaceStieltjes - " * string(N) * " recurrences on " * string(M) * " points.");
display(stieltjesresults["threaded-inplace"]);

println("GPUInplaceStieltjes - " * string(N) * " recurrences on " * string(M) * " points.");
display(stieltjesresults["GPU-inplace"]);

suite = BenchmarkGroup();

suite["fixed"] = @benchmarkable FixedLogKernel(N, x, ff) samples = 100 seconds = 500;
suite["inplace"] = @benchmarkable InplaceLogKernel(N, x, ff) samples = 100 seconds = 500;
suite["threaded-inplace"] = @benchmarkable ThreadedInplaceLogKernel(N, x, ff) samples = 100 seconds = 500;
suite["GPU-inplace"] = @benchmarkable GPUInplaceLogKernel(N, x, ff) samples = 100 seconds = 500;

logkernelresults = run(suite, verbose = true);

println("FixedLogKernel - " * string(N) * " recurrences on " * string(M) * " points.");
display(logkernelresults["fixed"]);

println("InplaceLogKernel - " * string(N) * " recurrences on " * string(M) * " points.");
display(logkernelresults["inplace"]);

println("ThreadedInplaceLogKernel - " * string(N) * " recurrences on " * string(M) * " points.");
display(logkernelresults["threaded-inplace"]);

println("GPUInplaceLogKernel - " * string(N) * " recurrences on " * string(M) * " points.");
display(logkernelresults["GPU-inplace"]);