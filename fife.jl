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
    allvalues::Array = allvalues,
)::VMState
    stackdepth = discrete.stackdepth
    programlen = discrete.programlen

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
    return state
    # NOTE if theres no stackpointer the discrete -> super -> discrete aren't consistent (eg symetric)
    # On the other hand super -> discrete is always an lossy process so it might not matter
end

function convert_continuous_to_discrete(
    contstate::VMState,
    allvalues = allvalues,
)::DiscreteVMState
    instructionpointer = onecold(contstate.instructionpointer)
    stackpointer = onecold(contstate.stackpointer)
    stack = [allvalues[i] for i in onecold(contstate.stack)]
    #variables = onecold(contstate.stack)

    stack = circshift(stack, 1 - stackpointer) # Check if this actually makes sense with circshift
    # Dealing with blanks is tricky. It's not clear what is correct semantically
    newstack = CircularDeque{StackValueType}(size(contstate.stack)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in stack
        if x == "blank"
            break
        else
            # TODO convert to int ?
            push!(newstack, x)
        end
    end
    return DiscreteVMState(; instructionpointer = instructionpointer, stack = newstack)
end

function ==(x::CircularDeque, y::CircularDeque)
    x.capacity != y.capacity && return false
    length(x) != length(y) && return false
    for (i, j) in zip(x, y)
        i == j || return false
    end
    return true
end

function ==(x::DiscreteVMState, y::DiscreteVMState)
    return x.instructionpointer == y.instructionpointer &&
           x.stack == y.stack &&
           x.variables == y.variables &&
           x.ishalted == y.ishalted
end

function ==(x::VMState, y::VMState)
    return x.instructionpointer == y.instructionpointer &&
           x.stackpointer == y.stackpointer &&
           x.stack == y.stack
end
