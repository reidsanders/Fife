using Random
using CUDA
using Flux
Random.seed!(123);
CUDA.allowscalar(false)

include("utils.jl")
using .Utils

if args.usegpu
    global device = gpu
    @info "Training on GPU"
else
    global device = cpu
    @info "Training on CPU"
end

StackFloatType = Float32
StackValueType = Int
largevalue = floor(StackValueType, sqrt(typemax(StackValueType)))

coercetostackvaluepart =
    partial(coercetostackvalue, StackValueType, -args.maxint, args.maxint, largevalue)

numericvalues = [[-largevalue]; [i for i = -args.maxint:args.maxint]; [largevalue]]
nonnumericvalues = ["blank"]
allvalues = [nonnumericvalues; numericvalues]
ishaltedvalues = [false, true]
blanks = fill("blank", args.stackdepth)
blankstack = onehotbatch(blanks, allvalues)
