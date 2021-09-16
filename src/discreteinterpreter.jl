# using Debugger
# using Base
import Base: +, -, *, length, convert, ==
# using BenchmarkTools
# using ProgressMeter
# using Base.Threads: @threads
using Parameters: @with_kw
# using Profile
using DataStructures: CircularDeque, CircularBuffer, Deque, DefaultDict
# using Test: @test

# include("utils.jl")
# StackValue = Int
StackFloatType = Float32

@with_kw struct StackValue
    val::Int = 0
    blank::Bool = true
    max::Bool = false
    min::Bool = false
    # OrderedPair(x1,x2,x3,x4) = x > y ? error("out of order") : new(x,y)
end
# StackValue(x) = StackValue(val = x)
function StackValue(x)
    if x >= args.maxint
        return StackValue(val=0,blank=false,max=true,min=false)
    elseif x <= -args.maxint
        return StackValue(val=0,blank=false,max=false,min=true)
    else
        return StackValue(val=x,blank=false,max=false,min=false)
    end
end

function +(x::StackValue, y::StackValue)
    if x.blank || y.blank
        return StackValue()
    elseif x.max
        if y.min 
            return StackValue(0)
        else
            return StackValue(blank=false, max=true)
        end
    elseif y.max
        if x.min 
            return StackValue(0)
        else
            return StackValue(blank=false, max=true)
        end
    elseif y.min || x.min
        return StackValue(blank=false, min=true)
    end

    StackValue(x.val + y.val)
end

function *(x::StackValue, y::StackValue)
    if x.blank || y.blank
        return StackValue()
    elseif x.max & y.min || x.min & y.max
        return StackValue(blank=false, min=true)
    elseif x.max & y.max || x.min & y.min
        return StackValue(blank=false, max=true)
    elseif x.max || y.max
        return StackValue(blank=false, max=true)
    elseif x.min || y.min
        return StackValue(blank=false, min=true)
    end

    StackValue(x.val * y.val)
end

function *(x::Number, y::StackValue)
    if y.blank
        return StackValue()
    elseif y.max & x > 0 || y.min & x < 0
        return StackValue(blank=false, max=true)
    elseif y.max & x < 0 || y.min & x > 0
        return StackValue(blank=false, min=true)
    end

    StackValue(x * y.val)
end
x::StackValue * y::Number = y * x

x::Number + y::StackValue = StackValue(x) + y
x::StackValue + y::Number = y + x
x::StackValue - y::StackValue = x + -1 * y
x::Number - y::StackValue = StackValue(x) - y
x::StackValue - y::Number = x - StackValue(y)


function ==(x::StackValue, y::StackValue)
    if y.blank && x.blank || y.max && x.max || y.min && x.min
        return true
    end
    x.val == y.val
end

x::StackValue == y::Number = x == StackValue(y)
x::Number == y::StackValue = StackValue(x) == y

function convert(::Type{StackValue}, x::Number)
    StackValue(x)
end


@with_kw mutable struct DiscreteVMState
    instrpointer::Int = 1
    input::CircularBuffer{StackValue} = CircularBuffer{StackValue}(args.inputlen)
    output::CircularBuffer{StackValue} = CircularBuffer{StackValue}(args.outputlen)
    stack::CircularBuffer{StackValue} = CircularBuffer{StackValue}(args.stackdepth)
    variables::DefaultDict{StackValue,StackValue} =
        DefaultDict{StackValue,StackValue}(0) # StackValue instead of Int?
    ishalted::Bool = false
    programlen::Int = args.programlen
    inputlen::Int = args.inputlen
    outputlen::Int = args.outputlen
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
    elseif targetinstrpointer > state.programlen
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

function instr_pushval!(value::StackValue, state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    pushfirst!(state.stack, value |> coercetostackvaluepart)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, y)
end

function instr_add!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x + y |> coercetostackvaluepart)
end

function instr_sub!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x - y |> coercetostackvaluepart)
end

function instr_mult!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    pushfirst!(state.stack, x * y |> coercetostackvaluepart)
end

function instr_div!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
        return
    end
    x = popfirst!(state.stack)
    y = popfirst!(state.stack)
    res = 1 * (x != 0 && y != 0)
    pushfirst!(state.stack, res)
end

function instr_or!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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
    if length(state.stack) < 1
        return
    elseif length(state.stack) < 2
        x = popfirst!(state.stack)
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

function instr_read!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.input) < 1
        return
    end
    x = popfirst!(state.input)
    pushfirst!(state.stack, x)
end

function instr_write!(state::DiscreteVMState)
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.stack) < 1
        return
    end
    x = popfirst!(state.stack)
    # setinstrpointer!(state, state.instrpointer + 1)
    pushfirst!(state.output, x)
end
