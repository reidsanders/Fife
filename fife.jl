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
    cont_instrpointer = onehot(discrete.instrpointer, [i for i = 1:programlen]) * 1.0f0

    discretestack = Array{Any,1}(undef, stackdepth)
    fill!(discretestack, "blank")
    for (i, x) in enumerate(discrete.stack)
        discretestack[i] = x
    end
    contstack = onehotbatch(discretestack, allvalues) * 1.0f0

    discretevariables = Array{Any,1}(undef, stackdepth)
    fill!(discretevariables, "blank")
    for (k, v) in discrete.variables
        discretevariables[k] = v
    end
    contvariables = onehotbatch(discretevariables, allvalues) * 1.0f0
    contishalted = onehot(discrete.ishalted, [false, true]) * 1.0f0
    VMState(
        cont_instrpointer |> device,
        contstate.stackpointer |> device,
        contstack |> device,
        contvariables |> device,
        contishalted |> device,
    )
end

function convert_continuous_to_discrete(
    contstate::VMState,
    allvalues = allvalues,
)::DiscreteVMState
    instrpointer = onecold(contstate.instrpointer)
    stackpointer = onecold(contstate.stackpointer)
    ishalted = onecold(contstate.ishalted, ishaltedvalues)
    stack = [allvalues[i] for i in onecold(contstate.stack)]
    variables = [allvalues[i] for i in onecold(contstate.variables)]
    #variables = onecold(contstate.stack)

    stack = circshift(stack, 1 - stackpointer) # Check if this actually makes sense with circshift
    # Dealing with blanks is tricky. It's not clear what is correct semantically
    newstack = CircularDeque{StackValueType}(size(contstate.stack)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in stack
        if x == "blank"
            break
            # or break... discrete can't have blank values on stack, but removing them 
            # is confusing and may mess up behavior if the superinterpreter is 
            # depending on there being a blank there
            # TODO either break or 
            # allow "blank" values on discrete stack? 
            # That would complicate the discrete operations a lot.
        else
            # TODO convert to int ?
            push!(newstack, x)
        end
    end

    newvariables = DefaultDict{StackValueType,StackValueType}(0)
    # default blank? blank isn't technically in it
    # Use missing instead of blank?
    for x in variables
        if x == "blank"
            continue
        else
            # TODO convert to int ?
            newvariables[]
            push!(newstack, x)
        end
    end
    DiscreteVMState(instrpointer = instrpointer, stack = newstack, ishalted = ishalted)
end

function ==(x::CircularDeque, y::CircularDeque)
    x.capacity != y.capacity && return false
    length(x) != length(y) && return false
    for (i, j) in zip(x, y)
        i == j || return false
    end
    true
end

function ==(x::DiscreteVMState, y::DiscreteVMState)
    x.instrpointer == y.instrpointer &&
        x.stack == y.stack &&
        x.variables == y.variables &&
        x.ishalted == y.ishalted
end

function ==(x::VMState, y::VMState)
    x.instrpointer == y.instrpointer &&
        x.stackpointer == y.stackpointer &&
        x.stack == y.stack
end
