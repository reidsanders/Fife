module DiscreteInterpreter
using Pkg
Pkg.activate(".")
using Debugger
using Base
import Base: +, -, *, length, convert
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

@with_kw mutable struct DiscreteVMState
    instructionpointer::Int = 1
    stack::Deque{Int} = Deque{Int}(args.stackdepth)
    variables::DefaultDict{Int,Int} = DefaultDict{Int,Int}(0)
    ishalted::Bool = false
end

function convert(::Type{Deque{T}}, x::Array{T,1}) where T
    y = Deque{T}()
    for el in x
        push!(y, el)
    end
    return y
end    

function instr_pass!(state::DiscreteVMState)
    state.instructionpointer += 1
end

function instr_halt!(state::DiscreteVMState)
    state.instructionpointer += 1
    state.ishalted = true
end

function instr_pushval!(value::Int, state::DiscreteVMState)
    state.instructionpointer += 1
    push!(state.stack, value)
end

val_instructions = [partial(instr_pushval!, i) for i in intvalues]

function instr_pop!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    pop!(state.stack)
end

function instr_dup!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    x = pop!(state.stack)
    push!(state.stack, x)
    push!(state.stack, x)
end

function instr_swap!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack)
    y = pop!(state.stack)
    push!(state.stack, x)
    push!(state.stack, y)
end

function instr_add!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x+y)
end

function instr_sub!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x-y)
end

function instr_mult!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x*y)
end

function instr_div!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    # Floor or Round?
    push!(state.stack, floor(x/y))
end

function instr_not!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    # 0 is false, anything else is true.
    # but if true still set to 1
    x = pop!(state.stack) 
    notx = 1 * (x != 0)
    push!(state.stack, notx)
end

function instr_and!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    res = 1 * (x!=0 && y!=0)
    push!(state.stack, res)
end

function instr_or!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    res = 1 * (x!=0 || y!=0)
    push!(state.stack, res)
end

function instr_goto!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    x = pop!(state.stack)
    # Verification of non zero, positive integer?
    if x > 0
        state.instructionpointer = x
    end
end

function instr_gotoif!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x != 0 && y > 0
        state.instructionpointer = y
    end
end

function instr_iseq!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x == y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
end

function instr_isgt!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x > y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
end

function instr_isge!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x >= y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
end

function instr_store!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = pop!(state.stack)
    y = pop!(state.stack)
    state.variables[y] = x
end

function instr_load!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.variables) < 1
        return
    end
    # TODO should this remove the address element on the stack or not
    if last(state.stack) in keys(state.variables)
        x = pop!(state.stack)
        push!(state.stack, state.variables[x])
    end
end

begin export
    DiscreteVMState,
    convert,
    instr_pass!,
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