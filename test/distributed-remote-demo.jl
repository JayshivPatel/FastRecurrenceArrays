using Distributed, Test;

# add remote processes created with Docker
addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2222", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/M4R")
addprocs(["root@localhost"]; tunnel=true, sshflags=["-p", "2223", "-o", "StrictHostKeyChecking=no"], exename="/usr/local/julia/bin/julia", dir="/tmp/M4R")

# activate the M4R environment 
@everywhere (import Pkg; Pkg.activate("."); Pkg.instantiate())

# load modules
@everywhere using FixedRecurrenceArrays, DistributedData;
@everywhere import RecurrenceRelationships: forwardrecurrence_next, forwardrecurrence_partial!;

# choose points inside the domain: (make complex)
z = (-5.0005:0.0001:5.0005) .+ 0*im;

# recurrence coefficients for Legendre
rec_P = (1:10), (1:2:10), -1 * (1:10);

# exact formula for Stieltjes transform of sqrt(1 - x²)
stieltjes_matrix = @. inv(z + sign(z) * sqrt(z^2-1));

y = DistributedFixedRecurrenceArray(z, rec_P, [stieltjes_matrix'; stieltjes_matrix'.^2], 1000);