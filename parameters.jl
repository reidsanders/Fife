using Random
using CUDA
using Flux
Random.seed!(123);
CUDA.allowscalar(false)

@with_kw mutable struct Args
    stackdepth::Int = 3
    programlen::Int = 3
    inputlen::Int = 1 # frozen part, assumed at front for now
    max_ticks::Int = 3
    maxint::Int = 10
    usegpu::Bool = false
end

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
