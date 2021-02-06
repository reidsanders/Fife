module DiscreteInterpreter
using Pkg
Pkg.activate(".")
using Debugger
using Base
import Base: +,-,*,length
using BenchmarkTools
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using Profile
using DataStructures: Deque, DefaultDict
using Flux: onehot, onehotbatch, onecold, crossentropy, logitcrossentropy, glorot_uniform, mse, epseltype
using Test: @test

#include("main.jl")
#using .SuperInterpreter: VMState

include("utils.jl")
using .Utils

@with_kw mutable struct Args
    stackdepth::Int = 10
    programlen::Int = 5
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 7
    usegpu::Bool = false
end

include("parameters.jl")

#=
function ==(x::Deque, y::Deque)
    for (i,el) in enumerate(x)
        if x[i] !== y[i]

end
=#

@with_kw mutable struct DiscreteVMState
    instructionpointer::Int = 1
    stack::Deque{Int} = Deque{Int}(args.stackdepth)
    variables::DefaultDict{Int,Int} = DefaultDict{Int,Int}(0)
    ishalted::Bool = false
end

instr_pass(state::DiscreteVMState) = state

function instr_halt!(state::DiscreteVMState)
    state.ishalted = true
    state.instructionpointer += 1
end

function instr_pushval!(value::Integer, state::DiscreteVMState)
    push!(state.stack, value)
    state.instructionpointer += 1
end

val_instructions = [partial(instr_pushval!, i) for i in intvalues]

function instr_pop!(state::DiscreteVMState)
    pop!(state.stack)
    state.instructionpointer += 1
end

function instr_dup!(state::DiscreteVMState)
    x = pop!(state.stack)
    push!(state.stack, x)
    push!(state.stack, x)
    state.instructionpointer += 1
end

function instr_swap!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    push!(state.stack, x)
    push!(state.stack, y)
    state.instructionpointer += 1
end

function instr_add!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x+y)
    state.instructionpointer += 1
end

function instr_sub!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x-y)
    state.instructionpointer += 1
end

function instr_mult!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x*y)
    state.instructionpointer += 1
end

function instr_div!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    # Floor or Round?
    push!(state.stack, floor(x/y))
    state.instructionpointer += 1
end

function instr_not!(state::DiscreteVMState)
    # 0 is false, anything else is true.
    # but if true still set to 1
    x = pop!(state.stack) 
    notx = 1 * (x != 0)
    push!(state.stack, notx)
    state.instructionpointer += 1
end

function instr_and!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    res = 1 * (x!=0 && y!=0)
    push!(state.stack, res)
    state.instructionpointer += 1
end

function instr_or!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    res = 1 * (x!=0 || y!=0)
    push!(state.stack, res)
    state.instructionpointer += 1
end

function instr_goto!(state::DiscreteVMState)
    x = pop!(state.stack)
    # Verification of non zero, positive integer?
    if x > 0
        state.instructionpointer = x
    else
        state.instructionpointer += 1
    end
end

function instr_gotoif!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x != 0 && y > 0
        state.instructionpointer = y
    else
        state.instructionpointer += 1
    end
end

function instr_iseq!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x == y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
    state.instructionpointer += 1
end

function instr_isgt!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x > y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
    state.instructionpointer += 1
end

function instr_isge!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x >= y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
    state.instructionpointer += 1
end

function instr_store!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    state.variables[y] = x
    state.instructionpointer += 1
end

function instr_load!(state::DiscreteVMState)
    # TODO should this remove the address element on the stack or not
    x = pop!(state.stack)
    push!(state.stack, state.variables[x])
    state.instructionpointer += 1
end

begin export
    DiscreteVMState,
    instr_halt!,
    instr_pushval!,
    instr_pop!,
    instr_dup!,
    instr_swap!,
    instr_add!,
    instr_sub!,
    instr_mult!,
    instr_div!,
    instr_not!,
    instr_and!,
    instr_goto!,
    instr_gotoif!,
    instr_iseq!,
    instr_isgt!,
    instr_isge!,
    instr_store!,
    instr_load!
end
end