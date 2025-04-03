using BenchmarkTools, Distributed;

# add a remote process created with Docker
addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2222", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");

# activate the FastRecurrenceArrays environment 
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());

# load modules
@everywhere using FastRecurrenceArrays;

# choose points
x = Float32.(-10.0:0.002:(10.0-0.002));

# num vectors
M = length(x);

# num recurrences
N = 1000;

# recurrence coefficients for ChebyshevU
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));

# parameters
params = (x, rec_U, N);

# distributed
matrix_result = @benchmarkable PartitionedRecurrenceArray(params...) samples = 100 seconds = 500;

function run_benchmark()
    matrix_display = run(matrix_result, verbose=true)
    println("PartitionedRecurrenceArray - " * string(N) * "×" * string(M))
    println("Workers: " * string(length(workers())))
    display(matrix_display)
end

#run_benchmark()

addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2223", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
@everywhere using FastRecurrenceArrays;

#run_benchmark()

addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2224", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
@everywhere using FastRecurrenceArrays;

#run_benchmark()

addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2225", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
@everywhere using FastRecurrenceArrays;

run_benchmark()