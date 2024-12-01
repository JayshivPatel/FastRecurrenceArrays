using FixedRecurrenceArrays, RecurrenceRelationshipArrays, 
    BenchmarkTools, BenchmarkPlots, StatsPlots, InfiniteArrays;

# choose points inside the domain: (make complex)
z_in = (-5.0005:0.001:5.0005) .+ 0*im;

# choose points outside the domain:
z_out = (1.0:0.001:100.0);

z = [z_in; z_out];

# recurrence coefficients for Legendre
rec_P = (1:10000), (1:2:10000), -1 * (1:10000);

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes = inv(z[1] + sign(z[1]) * sqrt(z[1]^2-1));
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2-1));

suite = BenchmarkGroup();

suite["Fixed"] = BenchmarkGroup(["vector", "matrix"])
suite["Threaded"] = BenchmarkGroup(["vector", "matrix"])

# fixed

suite["Fixed"]["vector"] = 
    @benchmarkable FixedRecurrenceArray(z[1], rec_P, [stieltjes, stieltjes^2], 10000) samples = 100000 evals = 1;
suite["Fixed"]["matrix"] = 
    @benchmarkable FixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2], 1000) samples = 1000 evals = 1;

# threaded

suite["Threaded"]["vector"] = 
    @benchmarkable ThreadedFixedRecurrenceArray(z[1], rec_P, [stieltjes, stieltjes^2], 10000) samples = 100000 evals = 1;
suite["Threaded"]["matrix"] = 
    @benchmarkable ThreadedFixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2], 1000) samples = 1000 evals = 1;

results = run(suite, verbose = true, seconds = 600);

# pretty-print the results

m_fixed_vec = mean(results["Fixed"]["vector"]);
m_fixed_matrix = mean(results["Fixed"]["matrix"]);

m_threaded_vec = mean(results["Threaded"]["vector"]);
m_threaded_matrix = mean(results["Threaded"]["matrix"]);

println("Vector Comparison:");
println(judge(m_threaded_vec, m_fixed_vec));

println("Matrix Comparison:");
println(judge(m_threaded_matrix, m_fixed_matrix));