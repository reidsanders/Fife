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
using Memoize

include("utils.jl")
using .Utils: partial

@with_kw struct VMState
    instructionpointer::Union{Array{Float32},CuArray{Float32}}
    stackpointer::Union{Array{Float32},CuArray{Float32}}
    stack::Union{Array{Float32},CuArray{Float32}}
    variables::Union{Array{Float32},CuArray{Float32}}
    ishalted::Union{Array{Float32},CuArray{Float32}}

    #=
    # Invariants here? Is it worth the extra calculation every construction though?
    function VMState(instructionpointer, stackpointer, stack, ishalted, variables)
        @assert isapprox(sum(state.instructionpointer), 1.0)
        @assert isapprox(sum(state.stackpointer), 1.0)
        for col in eachcol(state.stack)
            @assert isapprox(sum(col), 1.0)
        end
    end
    =#
end

struct VMSuperStates
    instructionpointers::Union{Array{Float32},CuArray{Float32}}
    stackpointers::Union{Array{Float32},CuArray{Float32}}
    stacks::Union{Array{Float32},CuArray{Float32}}
    supervariables::Union{Array{Float32},CuArray{Float32}}
    ishalteds::Union{Array{Float32},CuArray{Float32}}
end

a::Number * b::VMState = VMState(
    a * b.instructionpointer,
    a * b.stackpointer,
    a * b.stack,
    a * b.variables,
    a * b.ishalted,
)
a::VMState * b::Number = b * a
a::VMState + b::VMState = VMState(
    a.instructionpointer + b.instructionpointer,
    a.stackpointer + b.stackpointer,
    a.stack + b.stack,
    a.variables + b.variables,
    a.ishalted + b.ishalted,
)
a::VMState - b::VMState = VMState(
    a.instructionpointer - b.instructionpointer,
    a.stackpointer - b.stackpointer,
    a.stack - b.stack,
    a.variables - b.variables,
    a.ishalted - b.ishalted,
)

length(a::VMSuperStates) = size(a.instructionpointers)[3]
a::Union{Array,CuArray} * b::VMSuperStates = VMSuperStates(
    a .* b.instructionpointers,
    a .* b.stackpointers,
    a .* b.stacks,
    a .* b.supervariables,
    a .* b.ishalteds,
)
a::VMSuperStates * b::Union{Array,CuArray} = b * a

function super_step(state::VMState, program, instructions)
    # TODO instead of taking a state, take the separate arrays as args? Since CuArray doesn't like Structs
    # TODO batch the individual array (eg add superpose dimension -- can that be a struct or needs to be separate?)
    newstates = [instruction(state) for instruction in instructions]
    instructionpointers = cat([x.instructionpointer for x in newstates]..., dims = 3)
    stackpointers = cat([x.stackpointer for x in newstates]..., dims = 3)
    stacks = cat([x.stack for x in newstates]..., dims = 3)
    supervariables = cat([x.variables for x in newstates]..., dims = 3)
    ishalteds = cat([x.ishalted for x in newstates]..., dims = 3)

    states =
        VMSuperStates(instructionpointers, stackpointers, stacks, supervariables, ishalteds)
    current = program .* state.instructionpointer'
    summed = sum(current, dims = 2)
    summed = reshape(summed, (1, 1, :))
    scaledstates = summed * states
    reduced = VMState(
        sum(scaledstates.instructionpointers, dims = 3)[:, :, 1],
        sum(scaledstates.stackpointers, dims = 3)[:, :, 1],
        sum(scaledstates.stacks, dims = 3)[:, :, 1],
        sum(scaledstates.supervariables, dims = 3)[:, :, 1],
        sum(scaledstates.ishalteds, dims = 3)[:, :, 1],
    )
    return normit(reduced)
end

"""

advance instruction pointer respecting program length. Return newinstructionpointer, newishalted
If before the beginning of program set to 1, if after end set to end, and set ishalted to true
"""
function advanceinstructionpointer(state::VMState, increment::Int)
    maxallowedincrement = length(state.instructionpointer) - 1
    if increment == 0
        return (state.instructionpointer, state.ishalted)
    elseif increment > maxallowedincrement
        increment = maxallowedincrement
    elseif increment < - maxallowedincrement
        increment = 1 - maxallowedincrement
    end
    if increment > 0
        middle = state.instructionpointer[1: end - 1 - increment]
        middle = [zeros(abs(increment) - 1); middle]
    elseif increment < 0
        middle = state.instructionpointer[2 - increment: end]
        middle = [middle; zeros(abs(increment) - 1)]
    end
    plast = sum(state.instructionpointer[end-increment:end])
    pfirst = sum(state.instructionpointer[1:1-increment])
    newinstructionpointer = [[pfirst]; middle; [plast]]
    pfalse, ptrue = state.ishalted
    phalted = 1 - pfalse * (1 - plast) # 1 - pfalse + pfalse * plast
    newishalted = [1 - phalted, phalted]

    @assert sum(newinstructionpointer) ≈ 1.0 "Not sum to 1: $(newinstructionpointer)\n Initial: $(state.instructionpointer)"
    @assert sum(newishalted) ≈ 1.0

    return (newinstructionpointer .|> FloatType, newishalted .|> FloatType)
end

###############################
# Instructions
###############################

"""
    instr_dup!(state::VMState)::VMState

Get top of stack, then push it to stack. Return new state.

"""
function instr_dup!(state::VMState)::VMState
    #= 
    DUP Should duplicate top of stack, and push to top of stack 
    =#
    newstackpointer = circshift(state.stackpointer, -1)
    oldcomponent = state.stack .* (1 .- newstackpointer)'
    newcomponent = circshift(state.stack .* state.stackpointer', (0, -1))
    newstack = oldcomponent .+ newcomponent
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        newstackpointer,
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
    #= 
    ADD Should pop top two values, and add them, then push that value to top
    =#
    state, x = pop(state)
    state, y = pop(state)

    resultvec = op_probvec(+, x, y)
    newstate = push(state, resultvec)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        newstate.stackpointer,
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
    state, x = pop(state)
    state, y = pop(state)

    resultvec = op_probvec(-, x, y)
    newstate = push(state, resultvec)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        newstate.stackpointer,
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
    state, x = pop(state)
    state, y = pop(state)

    resultvec = op_probvec(*, x, y)
    newstate = push(state, resultvec)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        newstate.stackpointer,
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
    state, x = pop(state)
    state, y = pop(state)

    resultvec = op_probvec(/, x, y)
    newstate = push(state, resultvec)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        newstate.stackpointer,
        newstate.stack,
        state.variables,
        ishalted,
    )
end

"""
    instr_not!(state::VMState)::VMState

Pop value from stack, take not, then push result to stack. Return new state.

"""
function instr_not!(state::VMState)::VMState
    state, x = pop(state)

    resultvec = op_probvec(a -> float(a == 0), x)
    newstate = push(state, resultvec)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        newstate.stackpointer,
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
    state, x = pop(state)
    state, y = pop(state)

    state = push(state, x)
    state = push(state, y)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        state.stackpointer,
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
    state, conditional = pop(state)
    state, destination = pop(state)

    # Beginning of vector based approach to allow gpu (might be a big pain)
    # zerohotvec = onehot(0, allvalues)
    # probofgoto = 1 - sum((conditional .* zerohotvec))

    maxint = round(Int, (length(numericvalues) - 1) / 2)

    zeroindex = length(allvalues) - maxint
    neginfindex = zeroindex - maxint
    maxinstrindex = zeroindex + length(state.instructionpointer)

    probofgoto = 1 - conditional[zeroindex]
    gotobeginprob = sum(destination[neginfindex:zeroindex+1])
    jumpvalprobs = destination[zeroindex+1:maxinstrindex]
    gotoendprob = sum(destination[maxinstrindex:end])
    # TODO gotobegin set index 1.
    # TODO apply advanceinstructionpointer function instead of circshift?
    # TODO goto past end, adjust ishalted? (Or not?)
    currentinstructionforward = (1 - probofgoto) * circshift(state.instructionpointer, 1)
    newinstructionpointer = currentinstructionforward .+ probofgoto * jumpvalprobs
    # TODO ishalted += gotoendprob ?
    return VMState(
        newinstructionpointer,
        state.stackpointer,
        state.stack,
        state.variables,
        state.ishalted,
    )

end


"""
    instr_pass!(state::VMState)::VMState

Do nothing but advance instruction pointer. Returns new state.

"""
function instr_pass!(state::VMState)::VMState
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    return VMState(
        newinstructionpointer,
        state.stackpointer,
        state.stack,
        state.variables,
        ishalted,
    )
end


"""
    instr_pushval!(val::StackValueType, state::VMState)::VMState

Push current val to stack. Returns new state.

"""
function instr_pushval!(val::StackValueType, state::VMState)::VMState
    valhotvec = valhot(val, allvalues) # pass allvalues, and partial? 
    newstackpointer = circshift(state.stackpointer, -1)
    newinstructionpointer, ishalted = advanceinstructionpointer(state, 1)
    topscaled = valhotvec * newstackpointer'
    stackscaled = state.stack .* (1 .- newstackpointer')
    newstack = stackscaled .+ topscaled
    return VMState(
        newinstructionpointer,
        newstackpointer,
        newstack,
        state.variables,
        state.ishalted,
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
@memoize function optablepair(op; numericvalues = numericvalues)
    optable = op.(numericvalues, numericvalues')
    optable =
        coercetostackvalue.(optable, min = numericvalues[2], max = numericvalues[end-1])
    indexmapping = []
    for numericval in numericvalues
        append!(indexmapping, [findall(x -> x == numericval, optable)])
    end
    return indexmapping
end

"""
    optablesingle(op; [numericvalues = numericvalues])

Create table of all combinations of applying op to numericvalues. 
Find optable indexes that correspond to given value in numericvalues. Return this mapping.

"""
@memoize function optablesingle(op; numericvalues = numericvalues)
    optable = op.(numericvalues)
    optable =
        coercetostackvalue.(optable, min = numericvalues[2], max = numericvalues[end-1])
    indexmapping = []
    for numericval in numericvalues
        append!(indexmapping, [findall(x -> x == numericval, optable)])
    end
    return indexmapping
end

"""
    op_probvec(op, x::Array; numericvalues::Array = numericvalues)::Array

Apply numeric op to probability vector of mixed numeric and nonnumeric values. Returns new vector.

Requires numericvalues at end of allvalues.

"""
function op_probvec(op, x::Array; numericvalues::Array = numericvalues)::Array
    optableindexes = optablesingle(op, numericvalues = numericvalues)

    xnumerics = x[end+1-length(numericvalues):end]

    numericprobs = []
    for indexes in optableindexes
        append!(numericprobs, sum(xnumerics[indexes]))
    end
    nonnumericprobs = x[1:end-length(numericvalues)]


    @assert sum(xnumerics) ≈ sum(numericprobs) "Numeric probabilities are conserved"
    @assert sum(numericprobs) + sum(nonnumericprobs) ≈ 1 "Probabilities sum to one"

    return [nonnumericprobs; numericprobs]
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

    numericprobs = []
    for indexes in optableindexes
        append!(numericprobs, sum(probs[indexes]))
    end
    a = x[1:end-length(numericvalues)]
    b = y[1:end-length(numericvalues)]
    nonnumericprobs = a + b - a .* b

    @assert sum(xnumerics) * sum(ynumerics) ≈ sum(numericprobs) "Numeric probabilities are conserved"
    @assert sum(numericprobs) + sum(nonnumericprobs) ≈ 1 "Probabilities sum to one"

    return [nonnumericprobs; numericprobs]
end

"""
    pop(state::VMState; blankstack = blankstack)::Tuple(::VMState, ::Array)

Removes prob vector from stack. Returns the new state and top of stack.

"""
function pop(state::VMState; blankstack = blankstack)::Tuple{VMState,Array}
    scaledreturnstack = state.stack .* state.stackpointer'
    scaledremainingstack = state.stack .* (1 .- state.stackpointer')
    scaledblankstack = blankstack .* state.stackpointer'
    newstack = scaledremainingstack .+ scaledblankstack
    newstackpointer = circshift(state.stackpointer, 1)
    newstate = VMState(
        state.instructionpointer,
        newstackpointer,
        newstack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(newstate)
    return (newstate, dropdims(sum(scaledreturnstack, dims = 2), dims = 2))
end

"""
    push(state::VMState, valvec::Array)::VMState

Push prob vector to stack based on current stackpointer prob. Returns new state.

Note reversed arg ordering of instr in order to match regular push!

"""
function push(state::VMState, valvec::Array)::VMState
    @assert isapprox(sum(valvec), 1.0)
    newstackpointer = circshift(state.stackpointer, -1)
    topscaled = valvec * newstackpointer'
    stackscaled = state.stack .* (1 .- newstackpointer')
    newstack = stackscaled .+ topscaled
    newstate = VMState(
        state.instructionpointer,
        newstackpointer,
        newstack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(newstate)
    return newstate
end

function valhot(val, allvalues)
    return [i == val ? 1.0f0 : 0.0f0 for i in allvalues] |> device
end

function check_state_asserts(state::VMState)
    @assert sum(state.stackpointer) ≈ 1.0
    @assert sum(state.instructionpointer) ≈ 1.0
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
    @assert !any(isnan.(state.instructionpointer)) ## Damn, putting an assert removes the NaN
end

function softmaxmask(mask, prog)
    tmp = softmax(prog)
    trainable = tmp .* mask
    frozen = prog .* (1 .- mask)
    return trainable .+ frozen
end

function normit(a::Union{Array,CuArray}; dims = 1, ϵ = epseltype(a))
    new = a .+ ϵ
    return new ./ sum(new, dims = dims)
end

function normit(a::VMState; dims = 1)
    return VMState(
        normit(a.instructionpointer, dims = dims),
        normit(a.stackpointer, dims = dims),
        normit(a.stack, dims = dims),
        normit(a.variables, dims = dims),
        normit(a.ishalted, dims = dims),
    )
end

function main()
    state = init_state(stackdepth, programlen)
    state = run(state, program, instructions, max_ticks)
    return collapsed_program = onecold(program)
end

function applyfullmask(mask, prog)
    out = prog[mask]
    return reshape(out, (size(prog)[1], :))
end

###############################
# Initialization functions
###############################

function create_random_discrete_program(len, instructions)
    return program = [rand(instructions) for i = 1:len]
end

function create_random_inputs(len, instructions)
    return program = [rand(instructions) for i = 1:len]
end

function create_trainable_mask(programlen, inputlen)
    mask = falses(programlen)
    mask[inputlen+1:end] .= true
    return mask
end

function VMState(
    stackdepth::Int = args.stackdepth,
    programlen::Int = args.programlen,
    allvalues::Union{Array,CuArray} = allvalues,
)
    instructionpointer = zeros(Float32, programlen)
    stackpointer = zeros(Float32, stackdepth)
    ishalted = zeros(Float32, 2)
    stack = zeros(Float32, length(allvalues), stackdepth)
    variables = zeros(Float32, length(allvalues), stackdepth)
    instructionpointer[1] = 1.0
    stackpointer[1] = 1.0
    stack[1, :] .= 1.0
    variables[1, :] .= 1.0
    ishalted[1] = 1.0 # set false
    # @assert isbitstype(stack) == true
    state = VMState(
        instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
        variables |> device,
        ishalted |> device,
    )
    return state
end

function get_program_with_random_inputs(program, mask)
    num_instructions = size(program)[1]
    new_program = copy(program)
    for (i, col) in enumerate(eachcol(new_program[:, mask]))
        new_program[:, i] .= false
        new_program[rand(1:num_instructions), i] = true
    end
    return new_program
end

function create_program_batch(startprogram, trainmask, batch_size)
    target_program = program()
    for i = 1:batch_size
        program
    end
end

function create_examples(hiddenprogram, trainmaskfull; numexamples = 16)
    variablemasked = (1 .- trainmaskfull) .* hiddenprogram
    variablemaskeds = Array{Float32}(undef, (size(variablemasked)..., numexamples))
    for i = 1:numexamples
        newvariablemasked = copy(variablemasked)
        for col in eachcol(newvariablemasked)
            shuffle!(col)
        end
        variablemaskeds[:, :, i] = newvariablemasked  # Batch
    end
    return variablemaskeds
end

function run(state, program, instructions, ticks)
    for i = 1:ticks
        state = super_step(state, program, instructions)
        # assert_no_nans(state)
    end
    return state
end

function loss(ŷ, y)
    # TODO top of stack super position actual stack. add tiny amount of instructionpointer just because
    # Technically current instruction doesn't really matter
    return crossentropy(ŷ.stack, y.stack) +
           crossentropy(ŷ.stackpointer, y.stackpointer) +
           crossentropy(ŷ.instructionpointer, y.instructionpointer)
end

function accuracy(hidden, target, trainmask)
    samemax = onecold(hidden) .== onecold(target)
    return result = (sum(samemax) - sum(1 .- trainmask)) / sum(trainmask)
end

function test(hiddenprogram, targetprogram, blank_state, instructions, programlen)
    program = softmaxprog(hiddenprogram)
    target = run(blank_state, targetprogram, instructions, programlen)
    prediction = run(blank_state, program, instructions, programlen)
    return loss(prediction, target)
end

function forward(state, target, instructions, programlen, hiddenprogram)
    program = softmaxprog(hiddenprogram)
    pred = run(state, program, instructions, programlen)
    return loss(pred, target)
end

function trainloop(variablemaskeds; batchsize = 4)
    # TODO make true function without globals
    # (xbatch, ybatch)
    # grads = applyfullmaskprog(gradprog(data[1][1]))
    # hiddenprograms = varmasked .+ trainablemasked 
    # targetprograms = varmasked .+ targetmasked 
    grads = zeros(Float32, size(applyfullmaskprog(hiddenprogram)))
    @showprogress for i = 1:size(variablemaskeds)[3]
        newgrads = gradprog(hiddenprogram)
        grads = grads .+ applyfullmaskprog(newgrads)
        if i > 0 & i % batchsize == 0
            Optimise.update!(opt, trainablemasked, grads)
            grads .= 0
        end
    end
end

function trainloopsingle(hiddenprogram; numexamples = 4)
    # TODO make true function without globals
    @showprogress for i = 1:numexamples
        grads = gradprogpart(hiddenprogram)[end]
        grads = applyfullmasktohidden(grads)
        Optimise.update!(opt, hiddenprogram, grads)
    end
end

function trainbatch!(data; batchsize = 8)
    # TODO make true function without globals
    local training_loss
    grads = zeros(Float32, size(hiddenprogram[0]))
    @showprogress for d in data
        # TODO split hiddenprogram from data ?
        newgrads = gradprogpart(hiddenprogram)[end]
        grads = grads .+ applyfullmasktohidden(newgrads)
        if i > 0 & i % batchsize == 0
            Optimise.update!(opt, hiddenprogram, grads)
            grads .= 0
        end
    end
end

function normalize_stackpointer(state::VMState)
    stackpointermax = onecold(state.stackpointer)
    stack = circshift(state.stack, (0, 1 - stackpointermax))
    stackpointer = circshift(state.stackpointer, 1 - stackpointermax)
    return VMState(
        state.instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
        state.variables |> device,
        state.ishalted |> device,
    )
end
