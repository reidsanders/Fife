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
using Memoize

include("utils.jl")
using .Utils: partial

struct VMState
    instructionpointer::Union{Array{Float32},CuArray{Float32}}
    stackpointer::Union{Array{Float32},CuArray{Float32}}
    stack::Union{Array{Float32},CuArray{Float32}}
    # Invariants here?
end
struct VMSuperStates
    instructionpointers::Union{Array{Float32},CuArray{Float32}}
    stackpointers::Union{Array{Float32},CuArray{Float32}}
    stacks::Union{Array{Float32},CuArray{Float32}}
end

a::Number * b::VMState = VMState(a * b.instructionpointer, a * b.stackpointer, a * b.stack)
a::VMState * b::Number = b * a
a::VMState + b::VMState = VMState(
    a.instructionpointer + b.instructionpointer,
    a.stackpointer + b.stackpointer,
    a.stack + b.stack,
)
a::VMState - b::VMState = VMState(
    a.instructionpointer - b.instructionpointer,
    a.stackpointer - b.stackpointer,
    a.stack - b.stack,
)

length(a::VMSuperStates) = size(a.instructionpointers)[3]
a::Union{Array,CuArray} * b::VMSuperStates =
    VMSuperStates(a .* b.instructionpointers, a .* b.stackpointers, a .* b.stacks)
a::VMSuperStates * b::Union{Array,CuArray} = b * a

function super_step(state::VMState, program, instructions)
    # TODO instead of taking a state, take the separate arrays as args? Since CuArray doesn't like Structs
    # TODO batch the individual array (eg add superpose dimension -- can that be a struct or needs to be separate?)
    newstates = [instruction(state) for instruction in instructions]
    instructionpointers = cat([x.instructionpointer for x in newstates]..., dims = 3)
    stackpointers = cat([x.stackpointer for x in newstates]..., dims = 3)

    stacks = cat([x.stack for x in newstates]..., dims = 3)

    states = VMSuperStates(instructionpointers, stackpointers, stacks)
    current = program .* state.instructionpointer'
    summed = sum(current, dims = 2)
    summed = reshape(summed, (1, 1, :))
    scaledstates = summed * states
    reduced = VMState(
        sum(scaledstates.instructionpointers, dims = 3)[:, :, 1],
        sum(scaledstates.stackpointers, dims = 3)[:, :, 1],
        sum(scaledstates.stacks, dims = 3)[:, :, 1],
    )
    normit(reduced)
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
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstackpointer, newstack)

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
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstate.stackpointer, newstate.stack)
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
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstate.stackpointer, newstate.stack)
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
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstate.stackpointer, newstate.stack)
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
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstate.stackpointer, newstate.stack)
end

"""
    instr_not!(state::VMState)::VMState

Pop value from stack, take not, then push result to stack. Return new state.

"""
function instr_not!(state::VMState)::VMState
    state, x = pop(state)

    resultvec = op_probvec(!, x)
    newstate = push(state, resultvec)
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstate.stackpointer, newstate.stack)
end


@memoize function optablecalc(op; numericvalues = numericvalues)
    optable = op.(numericvalues, numericvalues')
    optable = replacenans.(optable, 0.0)
    optable =
        setoutofboundstoinf.(optable; min = numericvalues[2], max = numericvalues[end-1])

    indexmapping = []
    for numericval in numericvalues
        append!(indexmapping, [findall(x -> x == numericval, optable)])
    end
    indexmapping
end

"""
    op_probvec(op, x::Array, y::Array; numericvalues::Array = numericvalues)::Array

Apply op to probability vector of mixed numeric and nonnumeric values. Returns new vector.

Requires numericvalues at end of allvalues.

For non numeric values:
prob a is blank and b is blank + prob a is blank and b is not blank + prob b is blank and a is not blank?
a * b + a * (1-b) + b * (1-a) =>
ab + a - ab + b - ab =>
a + b - ab

"""
function op_probvec(op, x::Array, y::Array; numericvalues::Array = numericvalues)::Array
    optableindexes = optablecalc(op, numericvalues = numericvalues)

    xints = x[end+1-length(numericvalues):end]
    yints = y[end+1-length(numericvalues):end]
    probs = xints .* yints'

    numericprobs = []
    for indexes in optableindexes
        append!(numericprobs, sum(probs[indexes]))
    end
    a = x[1:end-length(numericvalues)]
    b = y[1:end-length(numericvalues)]
    nonnumericprobs = a + b - a .* b
    #@assert sum(xints) * sum(yints) ≈ sum(numericprobs)
    @test sum(xints) * sum(yints) ≈ sum(numericprobs)
    @test sum(numericprobs) ≈ 1 - sum(nonnumericprobs)
    
    [nonnumericprobs; numericprobs]
end

"""
    pop(state::VMState; blankstack = blankstack)::Tuple(::VMState, ::Array)

Removes prob vector from stack. Returns the new state and top of stack.

"""
function pop(state::VMState; blankstack = blankstack)::Tuple{VMState, Array}
    scaledreturnstack = state.stack .* state.stackpointer'
    scaledremainingstack = state.stack .* (1 .- state.stackpointer')
    scaledblankstack = blankstack .* state.stackpointer'
    newstack = scaledremainingstack .+ scaledblankstack
    newstackpointer = circshift(state.stackpointer, 1)
    newstate = VMState(state.instructionpointer, newstackpointer, newstack)
    check_state_asserts(newstate)
    (newstate, dropdims(sum(scaledreturnstack, dims = 2), dims = 2))
end

"""
    push(state::VMState, valvec::Array)::VMState

Push prob vector to stack based on current stackpointer prob. Returns new state.

Note reversed arg ordering of instr in order to match regular push!

"""
function push(state::VMState, valvec::Array)::VMState
    display(valvec)
    @assert isapprox(sum(valvec), 1.0)
    newstackpointer = circshift(state.stackpointer, -1)
    topscaled = valvec * newstackpointer'
    stackscaled = state.stack .* (1 .- newstackpointer')
    newstack = stackscaled .+ topscaled
    newstate = VMState(state.instructionpointer, newstackpointer, newstack)
    check_state_asserts(newstate)
    newstate
end

"""
    instr_swap!(state::VMState)::VMState

Swap top of stack with second to top of stack. Returns new state.

"""
function instr_swap!(state::VMState)::VMState
    newstackpointer = circshift(state.stackpointer, -1)
    newstack = state.stack .* (1 .- state.stackpointer') .+ state.stack .* newstackpointer'
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, newstackpointer, newstack)
end

"""
    instr_gotoiffull!(zerovec::Array, nonnumericvalues, state::VMState)::VMState

Pops top two elements of stack. If top is not zero, goto second element (or end, if greater than program len). Returns new state.

"""
function instr_gotoiffull!(zerovec::Array, nonnumericvalues, state::VMState)::VMState
    stackscaled = state.stack .* state.stackpointer'
    probofgoto = 1 .- sum(stackscaled .* zerovec, dims = 1)
    firstpoptop = circshift(state.stackpointer, 1)
    secondstackscaled = state.stack .* firstpoptop'
    jumpvalprobs = sum(secondstackscaled, dims = 2)

    # Assume ints start at 0 should skip 0?
    # Constraint -- length(numericvalues) > length of current instruction ?
    jumpvalprobs = jumpvalprobs[length(nonnumericvalues)+1:end]
    # TODO length(instructionpointer) compare and truncate
    jumpvalprobs = jumpvalprobs[1:end]
    currentinstructionforward =
        (1 - sum(jumpvalprobs)) * circshift(state.instructionpointer, 1)
    newinstructionpointer =
        currentinstructionforward .+ jumpvalprobs[1:length(state.instructionpointer)]
    newtop = circshift(firstpoptop, 1)

    # TODO Stack needs to be shifted too

    # jumpvalprobs[:end]
    # TODO set blank / zero to zero? zero goes to first? greater than length(program) goes to end?
    # for each value 0-max in stack, 
    # (??sum diagonal of equivalent jump locations. eg 0 at x is equivalent to 1 at x+1 on stack)
    # TODO if not int value, set to circshift 1 forward?
    # TODO add jumpvalprobs to instructionpointer? Then normalize?

    VMState(newinstructionpointer, newtop, state.stack)

end

"""
    instr_pass!(state::VMState)::VMState

Do nothing but advance instruction pointer. Returns new state.

"""
function instr_pass!(state::VMState)::VMState
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(newinstructionpointer, state.stackpointer, state.stack)
end


"""
    instr_pushval!(val::StackValueType, state::VMState)::VMState

Push current val to stack. Returns new state.

"""
function instr_pushval!(val::StackValueType, state::VMState)::VMState
    valhotvec = valhot(val, allvalues) # pass allvalues, and partial? 
    newstackpointer = circshift(state.stackpointer, -1)
    newinstructionpointer = circshift(state.instructionpointer, 1)
    topscaled = valhotvec * newstackpointer'
    stackscaled = state.stack .* (1 .- newstackpointer')
    newstack = stackscaled .+ topscaled
    VMState(newinstructionpointer, newstackpointer, newstack)
end

###############################
# Utility functions 
###############################

function valhot(val, allvalues)
    [i == val ? 1.0f0 : 0.0f0 for i in allvalues] |> device
end

function check_state_asserts(state)
    @assert isapprox(sum(state.instructionpointer), 1.0)
    @assert isapprox(sum(state.stackpointer), 1.0)
    for col in eachcol(state.stack)
        @assert isapprox(sum(col), 1.0)
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
    trainable .+ frozen
end

function normit(a::Union{Array,CuArray}; dims = 1, ϵ = epseltype(a))
    new = a .+ ϵ
    new ./ sum(new, dims = dims)
end

function normit(a::VMState; dims = 1)
    VMState(
        normit(a.instructionpointer, dims = dims),
        normit(a.stackpointer, dims = dims),
        normit(a.stack, dims = dims),
    )
end

function main()
    state = init_state(stackdepth, programlen)
    state = run(state, program, instructions, max_ticks)
    collapsed_program = onecold(program)
end

function applyfullmask(mask, prog)
    out = prog[mask]
    reshape(out, (size(prog)[1], :))
end

###############################
# Initialization functions
###############################

function create_random_discrete_program(len, instructions)
    program = [rand(instructions) for i = 1:len]
end

function create_random_inputs(len, instructions)
    program = [rand(instructions) for i = 1:len]
end

function create_trainable_mask(programlen, inputlen)
    mask = falses(programlen)
    mask[inputlen+1:end] .= true
    mask
end

function VMState(
    stackdepth::Int = args.stackdepth,
    programlen::Int = args.programlen,
    allvalues::Union{Array,CuArray} = allvalues,
)
    stack = zeros(Float32, length(allvalues), stackdepth)
    instructionpointer = zeros(Float32, programlen)
    stackpointer = zeros(Float32, stackdepth)
    stack[1, :] .= 1.0f0
    instructionpointer[1] = 1.0f0
    stackpointer[1] = 1.0f0
    # @assert isbitstype(stack) == true
    state = VMState(instructionpointer |> device, stackpointer |> device, stack |> device)
    state
end

function get_program_with_random_inputs(program, mask)
    num_instructions = size(program)[1]
    new_program = copy(program)
    for (i, col) in enumerate(eachcol(new_program[:, mask]))
        new_program[:, i] .= false
        new_program[rand(1:num_instructions), i] = true
    end
    new_program
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
    variablemaskeds
end

function run(state, program, instructions, ticks)
    for i = 1:ticks
        state = super_step(state, program, instructions)
        # assert_no_nans(state)
    end
    state
end

function loss(ŷ, y)
    # TODO top of stack super position actual stack. add tiny amount of instructionpointer just because
    # Technically current instruction doesn't really matter
    crossentropy(ŷ.stack, y.stack) +
    crossentropy(ŷ.stackpointer, y.stackpointer) +
    crossentropy(ŷ.instructionpointer, y.instructionpointer)
end

function accuracy(hidden, target, trainmask)
    samemax = onecold(hidden) .== onecold(target)
    result = (sum(samemax) - sum(1 .- trainmask)) / sum(trainmask)
end

function test(hiddenprogram, targetprogram, blank_state, instructions, programlen)
    program = softmaxprog(hiddenprogram)
    target = run(blank_state, targetprogram, instructions, programlen)
    prediction = run(blank_state, program, instructions, programlen)
    loss(prediction, target)
end

function forward(state, target, instructions, programlen, hiddenprogram)
    program = softmaxprog(hiddenprogram)
    pred = run(state, program, instructions, programlen)
    loss(pred, target)
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
    state =
        VMState(state.instructionpointer |> device, stackpointer |> device, stack |> device)
end
