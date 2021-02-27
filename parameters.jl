using Random
using CUDA
using Flux
Random.seed!(123);
CUDA.allowscalar(false)


if args.usegpu
    global device = gpu
    @info "Training on GPU"
else
    global device = cpu
    @info "Training on CPU"
end

numericvalues = [[-Inf]; [i for i = -args.maxint:args.maxint]; [Inf]]
nonnumericvalues = ["blank"]
allvalues = [nonnumericvalues; numericvalues]
blanks = fill("blank", args.stackdepth)
blankstack = onehotbatch(blanks, allvalues)
