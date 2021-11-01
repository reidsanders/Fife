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
StackFloatType = Float64

struct VMState
    instrpointer::Vector
    stackpointer::Array
    inputpointer::Array
    outputpointer::Array
    input::Matrix
    output::Matrix
    stack::Matrix
    variables::Matrix
    ishalted::Array # [nothalted, halted]
end

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

length(a::VMSuperStates) = size(a.instrpointers)[2]

a::Vector * b::VMSuperStates = VMSuperStates(
    permutedims(a) .* b.instrpointers,
    permutedims(a) .* b.stackpointers,
    permutedims(a) .* b.inputpointers,
    permutedims(a) .* b.outputpointers,
    reshape(a, (1, 1, :)) .* b.inputs,
    reshape(a, (1, 1, :)) .* b.outputs,
    reshape(a, (1, 1, :)) .* b.stacks,
    reshape(a, (1, 1, :)) .* b.supervariables,
    permutedims(a) .* b.ishalteds,
)

a::VMSuperStates * b::Union{Array,CuArray} = b * a

# function op_not(x::Number)::StackFloat
#     x == 0
# end

function op_not(x::StackValue)::StackValue
    if x.blank || x.max || x.min
        return 0
    end
    x.val == 0
end


function super_step(state::VMState, program, instructions)
    newstates = [instruction(state) for instruction in instructions]
    instrpointers = cat([x.instrpointer for x in newstates]..., dims = 2)
    stackpointers = cat([x.stackpointer for x in newstates]..., dims = 2)
    inputpointers = cat([x.inputpointer for x in newstates]..., dims = 2)
    outputpointers = cat([x.outputpointer for x in newstates]..., dims = 2)
    inputs = cat([x.input for x in newstates]..., dims = 3)
    outputs = cat([x.output for x in newstates]..., dims = 3)
    stacks = cat([x.stack for x in newstates]..., dims = 3)
    supervariables = cat([x.variables for x in newstates]..., dims = 3)
    ishalteds = cat([x.ishalted for x in newstates]..., dims = 2)

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
    currentprogram = program .* permutedims(state.instrpointer)
    summedprogram = dropdims(sum(currentprogram, dims = 2), dims = 2)
    scaledstates = summedprogram * states
    reduced = VMState(
        dropdims(sum(scaledstates.instrpointers, dims = 2), dims = 2),
        dropdims(sum(scaledstates.stackpointers, dims = 2), dims = 2),
        dropdims(sum(scaledstates.inputpointers, dims = 2), dims = 2),
        dropdims(sum(scaledstates.outputpointers, dims = 2), dims = 2),
        sum(scaledstates.inputs, dims = 3)[:, :, 1],
        sum(scaledstates.outputs, dims = 3)[:, :, 1],
        sum(scaledstates.stacks, dims = 3)[:, :, 1],
        sum(scaledstates.supervariables, dims = 3)[:, :, 1],
        dropdims(sum(scaledstates.ishalteds, dims = 2), dims = 2),
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

#TODO make ishalted merge state (if state ishalted then don't do anything? just return same state?
# (with ishalted = true so prob is correct))
# Make a decorator macro?
function applyishalted(a::VMState, b::VMState)::VMState
    # how to calc prob?
    VMState(
        a.instrpointer * a.ishalted[1] .+ b.instrpointer * a.ishalted[2],
        a.stackpointer * a.ishalted[1] .+ b.stackpointer * a.ishalted[2],
        a.inputpointer * a.ishalted[1] .+ b.inputpointer * a.ishalted[2],
        a.outputpointer * a.ishalted[1] .+ b.outputpointer * a.ishalted[2],
        a.input * a.ishalted[1] .+ b.input * a.ishalted[2],
        a.output * a.ishalted[1] .+ b.output * a.ishalted[2],
        a.stack * a.ishalted[1] .+ b.stack * a.ishalted[2],
        a.variables * a.ishalted[1] .+ b.variables * a.ishalted[2],
        b.ishalted, # TODO is this actually true?
    )
end

###############################
# Instructions
###############################

"""
    instr_dup!(state::VMState)::VMState

Get top of stack, then push it to stack. Return new state.

"""
function instr_pop!(state::VMState)::VMState
    state, x = popfromstack(state)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    applyishalted(
        state,
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
        ),
    )
end

"""
    instr_dup!(state::VMState)::VMState

Get top of stack, then push it to stack. Return new state.

"""
function instr_dup!(state::VMState)::VMState
    newstackpointer = circshift(state.stackpointer, -1)
    oldcomponent = state.stack .* permutedims((1 .- newstackpointer))
    newcomponent = circshift(state.stack .* permutedims(state.stackpointer), (0, -1))
    newstack = oldcomponent .+ newcomponent
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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

    resultvec = op_probvec((a, b) -> float(a != 0 && b != 0), x, y)
    newstate = pushtostack(state, resultvec)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    applyishalted(
        state,
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
        ),
    )
end

"""
    instr_swap!(state::VMState)::VMState

Swap top two values of stack. Returns new state.

"""
function instr_swap!(state::VMState)::VMState
    state, x = popfromstack(state)
    state, y = popfromstack(state)
    state = pushtostack(state, x)
    state = pushtostack(state, y)
    newinstrpointer, ishalted = advanceinstrpointer(state, 1)
    @assert isapprox(sum(newinstrpointer), 1, atol = 0.001) "instrpointer doesn't sum to 1: $(sum(newinstrpointer))\n $(newinstrpointer)\n Initial: $(state.instrpointer)"
    @assert isapprox(sum(ishalted), 1, atol = 0.001)
    applyishalted(
        state,
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
        ),
    )
end

"""
    instr_gotoif!(state::VMState; [allvalues=allvalues, numericvalues=numericvalues])::VMState

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
    ]
    destination = [
        [xb + yb - xb * yb]
        y[2:end] * ((1 - xb) * (1 - yb) + eps(yb)) / (1 - yb + eps(yb))
    ]
    @assert sum(x) ≈ 1
    @assert sum(y) ≈ 1

    ### Calc important indexes ###
    maxint = round(Int, (length(numericvalues) - 1) / 2)
    zeroindex = length(allvalues) - maxint
    neginfindex = zeroindex - maxint
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

    @assert p_ofgoto <= 1
    @assert p_gotobegin <= 1
    @assert p_gotoend <= 1
    @assert sum(jumpvalprobs) <= 1
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
    applyishalted(
        state,
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
        ),
    )
end

"""
    instr_goto!(state::VMState; [numericvalues=numericvalues])::VMState

Pops top two elements of stack. If top is not zero, goto second element (or end, if greater than program len). Returns new state.

"""
function instr_goto!(
    state::VMState;
    allvalues = allvalues,
    numericvalues = numericvalues,
)::VMState
    state, x = popfromstack(state)
    destination = x

    ### Calc important indexes ###
    maxint = round(Int, (length(numericvalues) - 1) / 2)
    zeroindex = length(allvalues) - maxint
    neginfindex = zeroindex - maxint
    maxinstrindex = zeroindex + length(state.instrpointer)

    ### Accumulate prob mass from goto off either end ###
    p_ofgoto = 1 - x[1]
    p_gotobegin = sum(destination[neginfindex:zeroindex+1])
    p_gotopastend = sum(destination[maxinstrindex+1:end])
    p_gotoend = destination[maxinstrindex] + p_gotopastend
    jumpvalprobs = destination[zeroindex+2:maxinstrindex-1]
    newinstrpointer = [[p_gotobegin]; jumpvalprobs; [p_gotoend]]

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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
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
    applyishalted(
        state,
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
        ),
    )
end


"""
    instr_pushval!(val::StackValue, state::VMState, allvalues::Array)::VMState

Push current val to stack. Returns new state.

"""
# global valhotvec = zeros(44) * 1.0f0
# valhotvec[20] = 1.0f0
function instr_pushval!(val::StackValue, state::VMState, allvalues::Array)::VMState
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
    applyishalted(
        state,
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
        ),
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
function optablepair(op, values)
    optable = op.(values, permutedims(values))
    [findall(x -> x == val, optable) for val in values]
end

"""
    optablesingle(op; [numericvalues = numericvalues])

Create table of all combinations of applying op to numericvalues. 
Find optable indexes that correspond to given value in numericvalues. Return this mapping.

"""
function optablesingle(op, values)
    optable = op.(values)
    [findall(x -> x == val, optable) for val in values]
end

"""
    op_probvec(op, x::Array; numericvalues::Array = numericvalues)::Array

Apply numeric op to probability vector of mixed numeric and nonnumeric values. Returns new vector.

Requires numericvalues at end of allvalues.

"""
function op_probvec(op, x::Array; values::Array = allvalues)
    @assert length(x) == length(values)
    optableindexes = optablesingle(op, values)
    probs = [sum(x[indexes]) for indexes in optableindexes]
    # @assert sum(probs) ≈ 1 "Probabilities don't sum to one: $(sum(probs)) != 1"
    probs
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
function op_probvec(op, x::Array, y::Array; values::Array = allvalues)::Array
    @assert length(x) == length(y)
    @assert length(x) == length(values)
    optableindexes = optablepair(op, values)
    opprobs = x .* permutedims(y)
    probs = [sum(opprobs[indexes]) for indexes in optableindexes]
    # @assert sum(probs) ≈ 1 "Probabilities don't sum to one: $(sum(probs)) != 1"
    probs
end

"""
    popfromstack(state::VMState; blankstack = blankstack)::Tuple(::VMState, ::Array)

Removes prob vector from stack. Returns the new state and top of stack.

"""
function popfromstack(state::VMState; blankstack = blankstack)::Tuple{VMState,Array}
    scaledreturnstack = state.stack .* permutedims(state.stackpointer)
    valvec = dropdims(sum(scaledreturnstack, dims = 2), dims = 2)
    scaledremainingstack = state.stack .* (1 .- permutedims(state.stackpointer))
    scaledblankstack = blankstack .* permutedims(state.stackpointer)
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
    scaledreturninput = state.input .* permutedims(state.inputpointer)
    valvec = dropdims(sum(scaledreturninput, dims = 2), dims = 2)
    scaledremaininginput = state.input .* (1 .- permutedims(state.inputpointer))
    scaledblankinput = blankinput .* permutedims(state.inputpointer)
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
    topscaled = valvec * permutedims(newstackpointer)
    stackscaled = state.stack .* (1 .- permutedims(newstackpointer))
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
    topscaled = valvec * permutedims(newoutputpointer)
    outputscaled = state.output .* (1 .- permutedims(newoutputpointer))
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
function fillinput(input::Array{StackValue}, inputlen::Int)::Array{StackFloatType}
    sinput = Array{StackValue,1}(undef, inputlen)
    fill!(sinput, StackValue())
    for (i, x) in enumerate(input)
        sinput[i] = x
    end
    onehotbatch(sinput, allvalues) * 1.0 # TODO |> StackFloat instead of * 1.0?
end

function fillinput(input::Array{Int}, inputlen::Int)::Array{StackFloatType}
    fillinput(StackValue.(input), inputlen)
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
    for col in eachcol(state.input)
        @assert sum(col) ≈ 1.0
    end
    for col in eachcol(state.output)
        @assert sum(col) ≈ 1.0
    end
    @assert all(x -> x >= 0, state.stack)
    @assert all(x -> x >= 0, state.stackpointer)
    @assert all(x -> x >= 0, state.instrpointer)
    @assert all(x -> x >= 0, state.variables)
    @assert all(x -> x >= 0, state.input)
    @assert all(x -> x >= 0, state.output)
    @assert all(x -> x >= 0, state.ishalted)
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
    state = runprogram(state, program, instructions, maxticks)
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
    variables = zeros(StackFloatType, length(allvalues), length(allvalues))
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

function normalize_iopointers(state::VMState)
    # Output pointer
    outputpointermax = onecold(state.outputpointer)
    output = circshift(state.output, (0, 1 - outputpointermax))
    outputpointer = circshift(state.outputpointer, 1 - outputpointermax)
    # Input pointer
    inputpointermax = onecold(state.inputpointer)
    input = circshift(state.input, (0, 1 - inputpointermax))
    inputpointer = circshift(state.inputpointer, 1 - inputpointermax)
    VMState(
        state.instrpointer |> device,
        state.stackpointer |> device,
        inputpointer |> device,
        outputpointer |> device,
        input |> device,
        output |> device,
        state.stack |> device,
        state.variables |> device,
        state.ishalted |> device,
    )
end
