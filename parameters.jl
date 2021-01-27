using Random
using CUDA
using Flux
Random.seed!(123);
CUDA.allowscalar(false)

args = Args()

if args.usegpu
    global device = gpu
    @info "Training on GPU"
else
    global device = cpu
    @info "Training on CPU"
end

intvalues = [i for i in 0:args.maxint]
nonintvalues = ["blank"]
allvalues = [nonintvalues; intvalues]
