import Base: +, -, *, length, convert, ==, show
using Parameters: @with_kw
using DataStructures: CircularDeque, CircularBuffer, Deque, DefaultDict
include("types.jl")
using .FifeTypes
import .FifeTypes: StackValue, show
# Set type based on arguments
function StackValue(x)
    StackValue(x, args.maxint, -args.maxint)
end

"""
    popfirstreplace!(x::CircularBuffer{StackValue})

Popfirst with blank value replacement at end.
"""
function popfirstreplace!(x::CircularBuffer{StackValue})
    item = popfirst!(x)
    push!(x, StackValue())
    return item
end

"""
    DiscreteVMState()

Create state for discrete stack based vm
"""
@with_kw mutable struct DiscreteVMState
    instrpointer::Int = 1
    input::CircularBuffer{StackValue} =
        fill!(CircularBuffer{StackValue}(args.inputlen), StackValue())
    output::CircularBuffer{StackValue} =
        fill!(CircularBuffer{StackValue}(args.outputlen), StackValue())
    stack::CircularBuffer{StackValue} =
        fill!(CircularBuffer{StackValue}(args.stackdepth), StackValue())
    variables::DefaultDict{StackValue,StackValue} =
        DefaultDict{StackValue,StackValue}(StackValue()) # StackValue instead of Int?
    ishalted::Bool = false
    programlen::Int = args.programlen
    inputlen::Int = args.inputlen
    outputlen::Int = args.outputlen
    stackdepth::Int = args.stackdepth
end

function DiscreteVMState(params)
    DiscreteVMState(
        1,
        fill!(CircularBuffer{StackValue}(params.inputlen), StackValue()),
        fill!(CircularBuffer{StackValue}(params.outputlen), StackValue()),
        fill!(CircularBuffer{StackValue}(params.stackdepth), StackValue()),
        DefaultDict{StackValue,StackValue}(StackValue()),
        false,
        params.programlen,
        params.inputlen,
        params.outputlen,
        params.stackdepth,
    )
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
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
end

function instr_halt!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    state.ishalted = true
end

function instr_pushval!(value::StackValue, state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    pushfirst!(state.stack, value)
end

instr_pushval!(val::Int, state::DiscreteVMState) = instr_pushval!(StackValue(val), state)

function instr_pop!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    popfirstreplace!(state.stack)
end

function instr_dup!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, x)
end

function instr_swap!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    pushfirst!(state.stack, x)
    pushfirst!(state.stack, y)
end

function instr_add!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    pushfirst!(state.stack, x + y)
end

function instr_sub!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    pushfirst!(state.stack, x - y)
end

function instr_mult!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    pushfirst!(state.stack, x * y)
end

function instr_div!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    # Floor or Round?
    pushfirst!(state.stack, x / y)
end

function instr_not!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    # 0 is false, anything else is true.
    # but if true still set to 1
    x = popfirstreplace!(state.stack)
    notx = 1 * (x == 0) # Replace with isnot
    pushfirst!(state.stack, notx)
end

function instr_and!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    res = 1 * (x != 0 && y != 0)
    pushfirst!(state.stack, res)
end

function instr_or!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    res = 1 * (x != 0 || y != 0)
    pushfirst!(state.stack, res)
end

function instr_goto!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    x = popfirstreplace!(state.stack)
    setinstrpointer!(state, x)
    # if x > 0
    #     setinstrpointer!(state, x)
    # else
    #     setinstrpointer!(state, state.instrpointer + 1)
    # end
end

function instr_gotoif!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    # if x == 0 || x.blank || y.blank
    if x == 0
        setinstrpointer!(state, state.instrpointer + 1)
    else
        setinstrpointer!(state, y)
    end
end

function instr_iseq!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    if x == y
        pushfirst!(state.stack, 1)
    else
        pushfirst!(state.stack, 0)
    end
end

function instr_isgt!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    if x > y
        pushfirst!(state.stack, 1)
    else
        pushfirst!(state.stack, 0)
    end
end

function instr_isge!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    if x >= y
        pushfirst!(state.stack, 1)
    else
        pushfirst!(state.stack, 0)
    end
end

function instr_store!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    y = popfirstreplace!(state.stack)
    state.variables[y] = x
end

function instr_load!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    if length(state.variables) < 1
        return
    end
    # TODO should this remove the address element on the stack or not
    if first(state.stack) in keys(state.variables)
        x = popfirstreplace!(state.stack)
        pushfirst!(state.stack, state.variables[x])
    end
end

function instr_read!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.input)
    pushfirst!(state.stack, x)
end

function instr_write!(state::DiscreteVMState)
    if state.ishalted
        return state
    end
    setinstrpointer!(state, state.instrpointer + 1)
    x = popfirstreplace!(state.stack)
    # setinstrpointer!(state, state.instrpointer + 1)
    pushfirst!(state.output, x)
end
