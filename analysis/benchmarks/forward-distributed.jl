using BenchmarkTools, Distributed;

# add a remote process created with Docker
addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2222", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
# activate the FastRecurrenceArrays environment 
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
# load module
@everywhere using FastRecurrenceArrays;

x = range(Float32(-10), Float32(10), 1_000_000);
M = length(x);
N = 10;
rec_U = (2 * ones(Float32, N), zeros(Float32, N), ones(Float32, N+1));
params = (x, rec_U, N);

function run_benchmark()
    println("PartitionedRecurrenceArray - " * string(N) * "×" * string(M))
    println("Workers: " * string(length(workers())))
    println(median(@benchmark PartitionedRecurrenceArray($params...) samples = 100 seconds = 500))
end

run_benchmark()

addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2223", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
@everywhere using FastRecurrenceArrays;
run_benchmark()

addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2224", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
@everywhere using FastRecurrenceArrays;
run_benchmark()

addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2225", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/FastRecurrenceArrays");
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate());
@everywhere using FastRecurrenceArrays;
run_benchmark()