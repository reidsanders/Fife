using Flux:
    onehot,
    onehotbatch,
    onecold,
    crossentropy,
    logitcrossentropy,
    glorot_uniform,
    mse,
    softmax,
    epseltype
using CUDA
import Base: +, -, *, length
using ProgressMeter
using Parameters: @with_kw
StackValueType = Int
StackFloatType = Float32
# VMArrayType =  ?

abstract type VM{T1} end
# struct VMState{T1 <: AbstractArray{<:AbstractFloat}} <: VM{T1}
#     instrpointer::T1
#     stackpointer::T1
#     stack::T1
#     variables::T1
#     ishalted::T1 # [nothalted, halted]
# end
struct VMState
    instrpointer::Array
    stackpointer::Array
    inputpointer::Array
    outputpointer::Array
    input::Array
    output::Array
    stack::Array
    variables::Array
    ishalted::Array # [nothalted, halted]
end


# VMState(a...) = VMState{Array{Float32}}(a...) # TODO pass?

# struct VMSuperStates{T1 <: AbstractArray{<:AbstractFloat}} <: VM{T1}
#     instrpointers::T1
#     stackpointers::T1
#     stacks::T1
#     supervariables::T1
#     ishalteds::T1
# end
struct VMSuperStates
    instrpointers::Array
    stackpointers::Array
    inputpointers::Array
    outputpointers::Array
    inputs::Array
    outputs::Array
    stacks::Array
    supervariables::Array
    ishalteds::Array
end

a::Number * b::VMState = VMState(
    a * b.instrpointer,
    a * b.stackpointer,
    a * b.inputpointer,
    a * b.outputpointer,
    a * b.input,
    a * b.output,
    a * b.stack,
    a * b.variables,
    a * b.ishalted,
)
a::VMState * b::Number = b * a

a::VMState + b::VMState = VMState(
    a.instrpointer + b.instrpointer,
    a.stackpointer + b.stackpointer,
    a.inputpointer + b.inputpointer,
    a.outputpointer + b.outputpointer,
    a.input + b.input,
    a.output + b.output,
    a.stack + b.stack,
    a.variables + b.variables,
    a.ishalted + b.ishalted,
)
a::VMState - b::VMState = VMState(
    a.instrpointer - b.instrpointer,
    a.stackpointer - b.stackpointer,
    a.inputpointer - b.inputpointer,
    a.outputpointer - b.outputpointer,
    a.input - b.input,
    a.output - b.output,
    a.stack - b.stack,
    a.variables - b.variables,
    a.ishalted - b.ishalted,
)

length(a::VMSuperStates) = size(a.instrpointers)[3]

a::Union{Array,CuArray} * b::VMSuperStates = VMSuperStates(
    a .* b.instrpointers,
    a .* b.stackpointers,
    a .* b.inputpointers,
    a .* b.outputpointers,
    a .* b.inputs,
    a .* b.outputs,
    a .* b.stacks,
    a .* b.supervariables,
    a .* b.ishalteds,
)

a::VMSuperStates * b::Union{Array,CuArray} = b * a

function op_not(x::Number)::StackFloatType
    x == 0
end

function super_step(state::VMState, program, instructions)
    newstates = [instruction(state) for instruction in instructions]
    instrpointers = cat([x.instrpointer for x in newstates]..., dims = 3)
    stackpointers = cat([x.stackpointer for x in newstates]..., dims = 3)
    inputpointers = cat([x.inputpointer for x in newstates]..., dims = 3)
    outputpointers = cat([x.outputpointer for x in newstates]..., dims = 3)
    inputs = cat([x.input for x in newstates]..., dims = 3)
    outputs = cat([x.output for x in newstates]..., dims = 3)
    stacks = cat([x.stack for x in newstates]..., dims = 3)
    supervariables = cat([x.variables for x in newstates]..., dims = 3)
    ishalteds = cat([x.ishalted for x in newstates]..., dims = 3)

    states = VMSuperStates(
        instrpointers,
        stackpointers,
        inputpointers,
        outputpointers,
        inputs,
        outputs,
        stacks,
        supervariables,
        ishalteds,
    )
    currentprogram = program .* state.instrpointer'
    summedprogram = sum(currentprogram, dims = 2)
    summedprogram = reshape(summedprogram, (1, 1, :))
    scaledstates = summedprogram * states
    reduced = VMState(
        sum(scaledstates.instrpointers, dims = 3)[:, :, 1],
        sum(scaledstates.stackpointers, dims = 3)[:, :, 1],
        sum(scaledstates.inputpointers, dims = 3)[:, :, 1],
        sum(scaledstates.outputpointers, dims = 3)[:, :, 1],
        sum(scaledstates.inputs, dims = 3)[:, :, 1],
        sum(scaledstates.outputs, dims = 3)[:, :, 1],
        sum(scaledstates.stacks, dims = 3)[:, :, 1],
        sum(scaledstates.supervariables, dims = 3)[:, :, 1],
        sum(scaledstates.ishalteds, dims = 3)[:, :, 1],
    )
    normit(reduced)
end

"""
    advanceinstrpointer(state::VMState, increment::Int)

advance instruction pointer respecting program length. Return (newinstrpointer, newishalted)
If before the beginning of program set to 1, if after end set to end, and set ishalted to true
"""
function advanceinstrpointer(state::VMState, increment::Int)
    maxincrement = length(state.instrpointer) - 1
    if increment == 0
        return (state.instrpointer, state.ishalted)
    elseif increment > maxincrement
        increment = maxincrement
    elseif increment < -maxincrement
        increment = 1 - maxincrement
    end
    if increment > 0
        middle = state.instrpointer[1:end-1-increment]
        middle = [zeros(abs(increment) - 1); middle]
    elseif increment < 0
        middle = state.instrpointer[2-increment:end]
        middle = [middle; zeros(abs(increment) - 1)]
    end
    p_pastend = sum(state.instrpointer[end+1-increment:end])
    p_begin = sum(state.instrpointer[1:1-increment])
    p_end = p_pastend + state.instrpointer[end-increment]
    newinstrpointer = [[p_begin]; middle; [p_end]]
    p_nothalted, p_halted = state.ishalted
    p_halted = 1 - p_nothalted * (1 - p_pastend)
    newishalted = [1 - p_halted, p_halted]

    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "Instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(newishalted), 1, atol = 0.001)

    # (newinstrpointer .|> StackFloatType, newishalted .|> StackFloatType)
    (newinstrpointer, newishalted)
end

###############################
# Instructions
###############################

"""
    instr_dup!(state::VMState)::VMState

Get top of stack, then push it to stack. Return new state.

"""
function instr_dup!(state::VMState)::VMState
    newstackpointer = circshift(state.stackpointer, -1)
    oldcomponent = state.stack .* (1 .- newstackpointer)'
    newcomponent = circshift(state.stack .* state.stackpointer', (0, -1))
    newstack = oldcomponent .+ newcomponent
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstack,
        state.variables,
        ishalted,
    )
end

"""
    instr_add!(state::VMState)::VMState

Pop two values from stack, add them, then push result to stack. Return new state.

"""
function instr_add!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)

    resultvec = op_probvec(+, x, y)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstate.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_sub!(state::VMState)::VMState

Pop two values of stack, subtract second from first, then push result to stack. Return new state.

"""
function instr_sub!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)

    resultvec = op_probvec(-, x, y)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstate.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_mult!(state::VMState)::VMState

Pop two values of stack, multiply them, then push result to stack. Return new state.

"""
function instr_mult!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)

    resultvec = op_probvec(*, x, y)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstate.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_div!(state::VMState)::VMState

Pop two values of stack, divide first by second, then push result to stack. Return new state.

"""
function instr_div!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)

    resultvec = op_probvec(/, x, y)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstate.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_not!(state::VMState)::VMState

Pop value from stack, apply not, then push result to stack. Return new state.

0 is false, all else is considered true. 

"""
function instr_not!(state::VMState)::VMState
    state, x = popfromstack(state)

    resultvec = op_probvec(op_not, x)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstate.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_and!(state::VMState)::VMState

Pop top two values from stack, apply and, then push result to stack. Return new state.

0 is false, all else is considered true. 

"""
function instr_and!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)

    resultvec = op_probvec((a, b) -> float(a != 0 && a != 0), x, y)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstate.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_swap!(state::VMState)::VMState

Swap top two values of stack. Returns new state.

"""
function instr_swap!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)

    xb = x[1]
    yb = y[1]
    x1 = xb + yb - xb * yb
    y1 = xb + yb - xb * yb
    xend = x[2:end] * ((1 - xb) * (1 - yb)) / (1 - xb + eps(xb))
    yend = y[2:end] * ((1 - xb) * (1 - yb)) / (1 - yb + eps(xb))
    xcomb = [x1; xend]
    ycomb = [y1; yend]

    state = pushtostack(state, xcomb)
    state = pushtostack(state, ycomb)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        state.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_gotoif!(state::VMState; [zerovec::Array = valhot(0, allvalues), nonnumericvalues=nonnumericvalues])::VMState

Pops top two elements of stack. If top is not zero, goto second element (or end, if greater than program len). Returns new state.

"""
function instr_gotoif!(
    state::VMState;
    allvalues = allvalues,
    numericvalues = numericvalues,
)::VMState
    check_state_asserts(state)
    state, x = popfromstack(state)
    state, y = popfromstack(state)
    xb = x[1]
    yb = y[1]
    conditional = [
        [xb + yb - xb * yb]
        x[2:end] * ((1 - xb) * (1 - yb) + eps(xb)) / (1 - xb + eps(xb))
    ] # Is it the eps making some things 1?
    destination = [
        [xb + yb - xb * yb]
        y[2:end] * ((1 - xb) * (1 - yb) + eps(yb)) / (1 - yb + eps(yb))
    ] # Is it the eps making some things 1?
    # x[2:end] = x[2:end] * ((1 - xb) * (1 - yb) + eps(xb)) / (1 - xb + eps(xb)) # Is it the eps making some things 1?
    # x[1] = xb + yb - xb * yb
    # y[2:end] = y[2:end] * ((1 - xb) * (1 - yb) + eps(xb)) / (1 - yb + eps(xb))
    # y[1] = xb + yb - xb * yb

    @assert sum(x) ≈ 1
    @assert sum(y) ≈ 1

    ### Calc important indexes ###
    maxint = round(Int, (length(numericvalues) - 3) / 2)
    zeroindex = length(allvalues) - maxint - 1
    neginfindex = zeroindex - maxint - 1
    maxinstrindex = zeroindex + length(state.instrpointer)

    ### Accumulate prob mass from goto off either end ###
    p_ofgoto =
        1 - (
            conditional[zeroindex] + conditional[1] + destination[1] -
            conditional[1] * destination[1]
        ) # include blank prob
    p_gotobegin = sum(destination[neginfindex:zeroindex+1])
    p_gotopastend = sum(destination[maxinstrindex+1:end])
    p_gotoend = destination[maxinstrindex] + p_gotopastend
    jumpvalprobs = destination[zeroindex+2:maxinstrindex-1]
    newinstrpointer = [[p_gotobegin]; jumpvalprobs; [p_gotoend]]

    # @assert p_ofgoto <= 1
    # @assert p_gotobegin <= 1
    # @assert p_gotoend <= 1
    # @assert sum(jumpvalprobs) <= 1
    # @show p_ofgoto
    # @show p_gotobegin
    # @show p_gotopastend
    # @show p_gotoend
    # @show jumpvalprobs
    ### calculate both nothing goto and just stepping forward
    currentinstructionforward, ishalted = advanceinstrpointer(state, 1)
    newinstrpointer =
        (1 - p_ofgoto) * currentinstructionforward .+ p_ofgoto * newinstrpointer

    p_nothalted, p_halted = state.ishalted
    p_halted = 1 - p_nothalted * (1 - p_gotopastend)
    newishalted = [1 - p_halted, p_halted]
    newinstrpointer = normit(newinstrpointer) # TODO may be covering up a bug
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.01) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(newishalted), 1, atol = 0.01)
    VMState(
        newinstrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        state.stack,
        state.variables,
        newishalted,
    )
end

"""
    instr_pass!(state::VMState)::VMState

Do nothing but advance instruction pointer. Returns new state.

"""
function instr_pass!(state::VMState)::VMState
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        state.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_pass!(state::VMState)::VMState

Do nothing but advance instruction pointer. Returns new state.

"""
function instr_halt!(state::VMState)::VMState
    newinstrpointer, _ = advanceinstrpointer(state, 1)
    ishalted = [0, 1]# .|> StackFloatType
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        state.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_read!(state::VMState)::VMState

Get input, then push to stack. Return new state.

"""
function instr_read!(state::VMState)::VMState
    state, x = popfrominput(state)
    state = pushtostack(state, x)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        state.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_write!(state::VMState)::VMState

Pop top value of stack, and push value to output. Return new state.

"""
function instr_write!(state::VMState)::VMState
    state, x = popfromstack(state)
    state = pushtooutput(state, x)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        state.stack,
        state.variables,
        ishalted,
    )
end


"""
    instr_pushval!(val::StackValueType, state::VMState, allvalues::Array)::VMState

Push current val to stack. Returns new state.

"""
# global valhotvec = zeros(44) * 1.0f0
# valhotvec[20] = 1.0f0
function instr_pushval!(val::StackValueType, state::VMState, allvalues::Array)::VMState
    # Verify that the mutation is coming from here
    # TODO either define a manual adjoint
    # TODO only use with partial, or try using BangBang? 
    # valhotvec = valhot(val, allvalues) # pass allvalues, and partial? 
    valhotvec = onehot(val, allvalues) * StackFloatType(1) # pass allvalues, and partial? 
    newstackpointer = circshift(state.stackpointer, -1)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    # newinstrpointer = circshift(state.instrpointer, 1)
    # ishalted = state.ishalted
    topscaled = valhotvec * newstackpointer'
    stackscaled = state.stack .* (1 .- newstackpointer')
    newstack = stackscaled .+ topscaled
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    VMState(
        newinstrpointer,
        newstackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstack,
        state.variables,
        ishalted,
    )
end

###############################
# Utility functions 
###############################
"""
    optablepair(op; [numericvalues = numericvalues])

Create table of all combinations of applying op to numericvalues. 
Find optable indexes that correspond to given value in numericvalues. Return this mapping.

"""
function optablepair(op; numericvalues = numericvalues)
    # function optablepair(op; numericvalues = numericvalues)
    optable = op.(numericvalues, numericvalues')
    optable = coercetostackvaluepart.(optable)
    indexmapping = [findall(x -> x == numericval, optable) for numericval in numericvalues]
end

"""
    optablesingle(op; [numericvalues = numericvalues])

Create table of all combinations of applying op to numericvalues. 
Find optable indexes that correspond to given value in numericvalues. Return this mapping.

"""
function optablesingle(op; numericvalues = numericvalues)
    optable = op.(numericvalues)
    optable = coercetostackvaluepart.(optable)
    indexmapping = [findall(x -> x == numericval, optable) for numericval in numericvalues]
end

"""
    op_probvec(op, x::Array; numericvalues::Array = numericvalues)::Array

Apply numeric op to probability vector of mixed numeric and nonnumeric values. Returns new vector.

Requires numericvalues at end of allvalues.

"""
function op_probvec(op, x::Array; numericvalues::Array = numericvalues)
    optableindexes = optablesingle(op, numericvalues = numericvalues)
    xnumerics = x[end+1-length(numericvalues):end]

    numericprobs = [sum(xnumerics[indexes]) for indexes in optableindexes]
    nonnumericprobs = x[1:end-length(numericvalues)]

    @assert sum(xnumerics) ≈ sum(numericprobs) "Numeric probabilities are conserved"
    @assert sum(numericprobs) + sum(nonnumericprobs) ≈ 1 "Probabilities sum to one"

    [nonnumericprobs; numericprobs]
end

"""
    op_probvec(op, x::Array, y::Array; numericvalues::Array = numericvalues)::Array

Apply numeric op to probability vector of mixed numeric and nonnumeric values. Returns new vector.

Requires numericvalues at end of allvalues.

For non numeric values:
prob a is blank and b is blank + prob a is blank and b is not blank + prob b is blank and a is not blank?
a * b + a * (1-b) + b * (1-a) =>
ab + a - ab + b - ab =>
a + b - ab

"""
function op_probvec(op, x::Array, y::Array; numericvalues::Array = numericvalues)::Array
    optableindexes = optablepair(op, numericvalues = numericvalues)
    xnumerics = x[end+1-length(numericvalues):end]
    ynumerics = y[end+1-length(numericvalues):end]
    probs = xnumerics .* ynumerics'

    numericprobs = [sum(probs[indexes]) for indexes in optableindexes]
    a = x[1:end-length(numericvalues)]
    b = y[1:end-length(numericvalues)]
    nonnumericprobs = a + b - a .* b

    @assert sum(xnumerics) * sum(ynumerics) ≈ sum(numericprobs) "Numeric probabilities are conserved"
    @assert sum(numericprobs) + sum(nonnumericprobs) ≈ 1 "Probabilities sum to one"

    [nonnumericprobs; numericprobs]
end

"""
    popfromstack(state::VMState; blankstack = blankstack)::Tuple(::VMState, ::Array)

Removes prob vector from stack. Returns the new state and top of stack.

"""
function popfromstack(state::VMState; blankstack = blankstack)::Tuple{VMState,Array}
    scaledreturnstack = state.stack .* state.stackpointer'
    valvec = dropdims(sum(scaledreturnstack, dims = 2), dims = 2)
    scaledremainingstack = state.stack .* (1 .- state.stackpointer')
    scaledblankstack = blankstack .* state.stackpointer'
    newstack = scaledremainingstack .+ scaledblankstack
    newstackpointer = circshift(state.stackpointer, 1)
    newstate = VMState(
        state.instrpointer,
        newstackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(newstate)
    (newstate, valvec)
end

"""
    popfrominput(state::VMState; blankinput = blankinput)::Tuple(::VMState, ::Array)

Removes prob vector from stack. Returns the new state and top of stack.

"""
function popfrominput(state::VMState; blankinput = blankinput)::Tuple{VMState,Array}
    scaledreturninput = state.input .* state.inputpointer'
    valvec = dropdims(sum(scaledreturninput, dims = 2), dims = 2)
    scaledremaininginput = state.input .* (1 .- state.inputpointer')
    scaledblankinput = blankinput .* state.inputpointer'
    newinput = scaledremaininginput .+ scaledblankinput
    newinputpointer = circshift(state.inputpointer, 1)
    newstate = VMState(
        state.instrpointer,
        state.stackpointer,
        newinputpointer,
        state.outputpointer,
        newinput,
        state.output,
        state.stack,
        state.variables,
        state.ishalted,
    )
    # newstate = pushtostack(newstate, valvec) #TODO edit all VMStates?
    check_state_asserts(newstate)
    (newstate, valvec)
end


"""
    pushtostack(state::VMState, valvec::Array)::VMState

Push prob vector to stack based on current stackpointer prob. Returns new state.

Note reversed arg ordering of instr in order to match regular push!

"""
function pushtostack(state::VMState, valvec::Array)::VMState
    @assert isapprox(sum(valvec), 1.0) "Value vector doesn't sum to 1: $(sum(valvec))"
    newstackpointer = circshift(state.stackpointer, -1)
    topscaled = valvec * newstackpointer'
    stackscaled = state.stack .* (1 .- newstackpointer')
    newstack = stackscaled .+ topscaled
    newstate = VMState(
        state.instrpointer,
        newstackpointer,
        state.inputpointer,
        state.outputpointer,
        state.input,
        state.output,
        newstack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(newstate)
    newstate
end

"""
    pushtooutput(state::VMState, valvec::Array)::VMState

Push prob vector to output based on current outputpointer prob. Returns new state.

Note reversed arg ordering of instr in order to match regular push!

"""
function pushtooutput(state::VMState, valvec::Array)::VMState
    @assert isapprox(sum(valvec), 1.0) "Value vector doesn't sum to 1: $(sum(valvec))"
    newoutputpointer = circshift(state.outputpointer, -1)
    topscaled = valvec * newoutputpointer'
    outputscaled = state.output .* (1 .- newoutputpointer')
    newoutput = outputscaled .+ topscaled
    newstate = VMState(
        state.instrpointer,
        state.stackpointer,
        state.inputpointer,
        newoutputpointer,
        state.input,
        newoutput,
        state.stack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(newstate)
    newstate
end

"""
    fillinput(state::VMState; blankinput = blankinput)::Tuple{VMState,Array}

Populates onehot input stack from array. Returns the input stack.

"""
function fillinput(input::Array{StackValueType}, inputlen::Int)::Array{StackFloatType}
    sinput = Array{Any,1}(undef, inputlen)
    fill!(sinput, "blank")
    for (i, x) in enumerate(input)
        sinput[i] = x
    end
    onehotbatch(sinput, allvalues) * 1.0
end

function valhot(val, allvalues)
    [i == val ? 1.0f0 : 0.0f0 for i in allvalues] |> device
end

function check_state_asserts(state::VMState)
    @assert sum(state.stackpointer) ≈ 1.0
    @assert sum(state.instrpointer) ≈ 1.0
    @assert sum(state.ishalted) ≈ 1.0
    for col in eachcol(state.stack)
        @assert sum(col) ≈ 1.0
    end
    for col in eachcol(state.variables)
        @assert sum(col) ≈ 1.0
    end
end

function assert_no_nans(state::VMState)
    @assert !any(isnan.(state.stack)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.stackpointer)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.instrpointer)) ## Damn, putting an assert removes the NaN
end

function softmaxmask(mask, prog)
    tmp = softmax(prog)
    trainable = tmp .* mask
    frozen = prog .* (1 .- mask)
    trainable .+ frozen
end

function normit(a::Union{Array,CuArray}; dims = 1, ϵ = epseltype(a))
    new = a .+ ϵ
    new ./ sum(new, dims = dims)
end

function normit(a::VMState; dims = 1)
    VMState(
        normit(a.instrpointer, dims = dims),
        normit(a.stackpointer, dims = dims),
        normit(a.inputpointer, dims = dims),
        normit(a.outputpointer, dims = dims),
        normit(a.input, dims = dims),
        normit(a.output, dims = dims),
        normit(a.stack, dims = dims),
        normit(a.variables, dims = dims),
        normit(a.ishalted, dims = dims),
    )
end

function main()
    state = init_state(stackdepth, programlen)
    state = runprogram(state, program, instructions, max_ticks)
    collapsed_program = onecold(program)
end

function applyfullmask(mask, prog)
    out = prog[mask]
    reshape(out, (size(prog)[1], :))
end

###############################
# Initialization functions
###############################

function VMState(
    stackdepth::Int = args.stackdepth,
    programlen::Int = args.programlen,
    allvalues::AbstractArray = allvalues,
    inputlen::Int = args.inputlen,
    outputlen::Int = args.outputlen,
)
    instrpointer = zeros(StackFloatType, programlen)
    stackpointer = zeros(StackFloatType, stackdepth)
    inputpointer = zeros(StackFloatType, inputlen)
    outputpointer = zeros(StackFloatType, outputlen)
    ishalted = zeros(StackFloatType, 2)
    stack = zeros(StackFloatType, length(allvalues), stackdepth)
    input = zeros(StackFloatType, length(allvalues), inputlen)
    output = zeros(StackFloatType, length(allvalues), outputlen)
    variables = zeros(StackFloatType, length(allvalues), stackdepth)
    instrpointer[1] = 1.0
    stackpointer[1] = 1.0
    inputpointer[1] = 1.0
    outputpointer[1] = 1.0
    stack[1, :] .= 1.0
    input[1, :] .= 1.0
    output[1, :] .= 1.0
    variables[1, :] .= 1.0
    ishalted[1] = 1.0 # set false
    VMState(
        instrpointer |> device,
        stackpointer |> device,
        inputpointer |> device,
        outputpointer |> device,
        input |> device,
        output |> device,
        stack |> device,
        variables |> device,
        ishalted |> device,
    )
end

function normalize_stackpointer(state::VMState)
    stackpointermax = onecold(state.stackpointer)
    stack = circshift(state.stack, (0, 1 - stackpointermax))
    stackpointer = circshift(state.stackpointer, 1 - stackpointermax)
    VMState(
        state.instrpointer |> device,
        stackpointer |> device,
        state.inputpointer |> device,
        state.outputpointer |> device,
        state.input |> device,
        state.output |> device,
        stack |> device,
        state.variables |> device,
        state.ishalted |> device,
    )
end
