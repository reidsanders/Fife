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
#=
@with_kw mutable struct Args
    stackdepth::Int = 10
    programlen::Int = 5
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 7
    usegpu::Bool = false
end
=#

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
    pushfirst!(state.stack, value)
end

val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

function instr_pop!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    popfirst!(state.stack)
end

function instr_dup!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    x = popfirst!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, x)
    return state
end

function instr_swap!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, y)
end

function instr_add!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack) 
    y = popfirst!(state.stack) 
    pushfirst!(state.stack, x+y)
end

function instr_sub!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack) 
    y = popfirst!(state.stack) 
    pushfirst!(state.stack, x-y)
end

function instr_mult!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack) 
    y = popfirst!(state.stack) 
    pushfirst!(state.stack, x*y)
end

function instr_div!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack) 
    y = popfirst!(state.stack) 
    # Floor or Round?
    pushfirst!(state.stack, floor(x/y))
end

function instr_not!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    # 0 is false, anything else is true.
    # but if true still set to 1
    x = popfirst!(state.stack) 
    notx = 1 * (x != 0)
    pushfirst!(state.stack, notx)
end

function instr_and!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack) 
    y = popfirst!(state.stack) 
    res = 1 * (x!=0 && y!=0)
    pushfirst!(state.stack, res)
end

function instr_or!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack) 
    y = popfirst!(state.stack) 
    res = 1 * (x!=0 || y!=0)
    pushfirst!(state.stack, res)
end

function instr_goto!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 1
        return
    end
    x = popfirst!(state.stack)
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
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    if x != 0 && y > 0
        state.instructionpointer = y
    end
end

function instr_iseq!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    if x == y
        pushfirst!(state.stack, 1)
    else
        pushfirst!(state.stack, 0)
    end
end

function instr_isgt!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    if x > y
        pushfirst!(state.stack, 1)
    else
        pushfirst!(state.stack, 0)
    end
end

function instr_isge!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    if x >= y
        pushfirst!(state.stack, 1)
    else
        pushfirst!(state.stack, 0)
    end
end

function instr_store!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    state.variables[y] = x
end

function instr_load!(state::DiscreteVMState)
    state.instructionpointer += 1
    if length(state.variables) < 1
        return
    end
    # TODO should this remove the address element on the stack or not
    if last(state.stack) in keys(state.variables)
        x = popfirst!(state.stack)
        pushfirst!(state.stack, state.variables[x])
    end
end

#=
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
=#
end