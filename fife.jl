using Pkg
Pkg.activate(".")
using Flux
using Flux:
    onehot,
    onehotbatch,
    onecold,
    crossentropy,
    logitcrossentropy,
    glorot_uniform,
    mse,
    epseltype
using Flux: Optimise
using CUDA
using Zygote
using Random
using LoopVectorization
using Debugger
using Base
import Base: +, -, *, length
using StructArrays
using BenchmarkTools
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using Profile
using DataStructures: Deque

@with_kw mutable struct Args
    stackdepth::Int = 10
    programlen::Int = 10
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 20
    usegpu::Bool = false
end
args = Args()
include("parameters.jl")
print(args)
#end

include("utils.jl")
using .Utils

include("discreteinterpreter.jl")
include("superinterpreter.jl")

val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

#########################
#     Conversion        #
#########################
function convert_discrete_to_continuous(
    discrete::DiscreteVMState,
    stackdepth = args.stackdepth,
    programlen = args.programlen,
    allvalues::Array = allvalues,
)::VMState
    contstate = VMState(stackdepth, programlen, allvalues)
    cont_instructionpointer =
        onehot(discrete.instructionpointer, [i for i = 1:programlen]) * 1.0f0
    discretestack = Array{Any,1}(undef, stackdepth)
    fill!(discretestack, "blank")

    for (i, x) in enumerate(discrete.stack)
        discretestack[i] = x
    end
    cont_stack = onehotbatch(discretestack, allvalues) * 1.0f0
    state = VMState(
        cont_instructionpointer |> device,
        contstate.stackpointer |> device,
        cont_stack |> device,
    )
    state
    # NOTE if theres no stackpointer the discrete -> super -> discrete aren't consistent (eg symetric)
    # On the other hand super -> discrete is always an lossy process
end

function convert_continuous_to_discrete(
    contstate::VMState,
    stackdepth = args.stackdepth,
    programlen = args.programlen,
    allvalues = allvalues,
)::DiscreteVMState
    instructionpointer = onecold(contstate.instructionpointer)
    stackpointer = onecold(contstate.stackpointer)
    stack = [allvalues[i] for i in onecold(contstate.stack)]
    #variables = onecold(contstate.stack)

    stack = circshift(stack, 1 - stackpointer) # Check if this actually makes sense with circshift
    # Dealing with blanks is tricky. It's not clear what is correct semantically
    newstack = Vector{StackValueType}() # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in stack
        if x == "blank"
            break
        else
            push!(newstack, x)
        end
    end
    DiscreteVMState(; instructionpointer = instructionpointer, stack = newstack)
end
