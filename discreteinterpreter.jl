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
using DataStructures: CircularDeque, Deque, DefaultDict
using Test: @test

include("utils.jl")
using .Utils: partial, replacenans, setoutofboundstoinf, roundnoninf, coercetostackvalue

coercetostackvaluepart(x) = coercetostackvalue(x; min = -args.maxint, max = args.maxint)

StackValueType = Real

# Set nans to 0
# Set > max to Inf
# Set < max to -Inf
# Round Floats to Int (?) define convert? probably want to only clamp at end of calculations

@with_kw mutable struct DiscreteVMState
    instrpointer::Int = 1
    stack::CircularDeque{StackValueType} = CircularDeque{StackValueType}(args.stackdepth)
    variables::DefaultDict{Int,Int} = DefaultDict{Int,Int}(0) # StackValueType instead of Int?
    ishalted::Bool = false
    programlen::Int = args.programlen
    stackdepth::Int = args.stackdepth
end

function convert(::Type{CircularDeque{T}}, x::Array{T,1}) where {T}
    y = CircularDeque{T}(length(x))
    for el in x
        push!(y, el)
    end
    y
end

"""
    setinstrpointer(state::DiscreteVMState, newinstrpointer)

Set instruction pointer respecting program length. 
If before the beginning of program set to 1, if after end set to end, and set ishalted to true
"""
function setinstrpointer!(state::DiscreteVMState, targetinstrpointer)
    if targetinstrpointer < 1
        state.instrpointer = 1
    elseif targetinstrpointer >= state.programlen
        state.instrpointer = state.programlen
        state.ishalted = true
    else
        state.instrpointer = targetinstrpointer
    end
end


function instr_pass!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
end

function instr_halt!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    state.ishalted = true
end

function instr_pushval!(value::StackValueType, state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    pushfirst!(state.stack, value)
end

function instr_pop!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    end
    popfirst!(state.stack)
end

function instr_dup!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    end
    x = popfirst!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, x)
    state
end

function instr_swap!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, y)
end

function instr_add!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x + y |> coercetostackvaluepart)
end

function instr_sub!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x - y |> coercetostackvaluepart)
end

function instr_mult!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x * y |> coercetostackvaluepart)
end

function instr_div!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    # Floor or Round?
    pushfirst!(state.stack, x / y |> coercetostackvaluepart)
end

function instr_not!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    end
    # 0 is false, anything else is true.
    # but if true still set to 1
    x = popfirst!(state.stack)
    notx = 1 * (x == 0)
    pushfirst!(state.stack, notx)
end

function instr_and!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    res = 1 * (x != 0 && y != 0)
    pushfirst!(state.stack, res)
end

function instr_or!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    res = 1 * (x != 0 || y != 0)
    pushfirst!(state.stack, res)
end

function instr_goto!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    end
    x = popfirst!(state.stack)
    # Verification of non zero, positive integer?
    if x > 0
        state.instrpointer = x
    end
end

function instr_gotoif!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    if x != 0 && y > 0
        # TODO clamp to valid length
        state.instrpointer = y
    end
end

function instr_iseq!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
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
    setinstrpointer!(state, state.instrpointer + 1)
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
    setinstrpointer!(state, state.instrpointer + 1)
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
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 2
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    state.variables[y] = x
end

function instr_load!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.variables) < 1
        return
    end
    # TODO should this remove the address element on the stack or not
    if last(state.stack) in keys(state.variables)
        x = popfirst!(state.stack)
        pushfirst!(state.stack, state.variables[x])
    end
end
