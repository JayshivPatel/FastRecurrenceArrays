import CUDA: has_cuda, has_cuda_gpu

include("basictests.jl");
include("integraltests.jl");

# Run GPU unit tests if GPU present
if has_cuda() && has_cuda_gpu()
    include("gputests.jl");
end

# Run threaded tests if Julia was started with more than one thread
if Threads.nthreads() > 1
    include("threadedtests.jl");
end