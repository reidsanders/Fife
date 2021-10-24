module Fife

using Base: Bool
using Flux
using Flux:
    onehot,
    onehotbatch,
    onecold,
    crossentropy,
    logitcrossentropy,
    glorot_uniform,
    mse,
    epseltype,
    cpu,
    gpu,
    Optimise,
    gradient

using Memoize
import Base: +, -, *, length, ==
using Parameters: @with_kw
include("utils.jl")
include("discreteinterpreter.jl")
# include("types.jl")
# using .FifeTypes
#TODO remove mutable / make const?
@with_kw mutable struct Args
    batchsize::Int = 2
    lr::Float64 = 1e-3
    epochs::Int = 2
    stackdepth::Int = 11
    programlen::Int = 13
    inputlen::Int = 7
    outputlen::Int = 8
    maxticks::Int = 100
    maxint::Int = 20
    trainsetsize::Int = 10
    usegpu::Bool = false
    StackFloatType::Type = Float64
end
args = Args()

function create_dependent_values(args)
    if args.usegpu
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    # numericvalues = [[-args.maxint]; [i for i = -args.maxint:args.maxint]; [args.maxint]]
    numericvalues = StackValue.([i for i = -args.maxint:args.maxint])
    nonnumericvalues = [StackValue()]
    allvalues = StackValue.([nonnumericvalues; numericvalues])
    ishaltedvalues = [false, true] #TODO StackValue?
    blanks = fill(StackValue(), args.stackdepth)
    blankstack = onehotbatch(blanks, allvalues)
    inputblanks = fill(StackValue(), args.inputlen)
    blankinput = onehotbatch(inputblanks, allvalues)
    outputblanks = fill(StackValue(), args.outputlen)
    blankoutput = onehotbatch(outputblanks, allvalues)

    (
        device,
        numericvalues,
        nonnumericvalues,
        allvalues,
        ishaltedvalues,
        blanks,
        blankstack,
        blankinput,
        blankoutput,
    )
end

device,
numericvalues,
nonnumericvalues,
allvalues,
ishaltedvalues,
blanks,
blankstack,
blankinput,
blankoutput = create_dependent_values(args)

## Set maxint for StackValue 
MAXINT = args.maxint

include("superinterpreter.jl")

begin
    export partial,
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
        instr_load!,
        instr_read!,
        instr_write!,
        VMState,
        DiscreteVMState,
        convert_continuous_to_discrete,
        convert_discrete_to_continuous,
        create_dependent_values,
        normit,
        fillinput,
        check_state_asserts,
        ==,
        runprogram,
        loss,
        accuracy,
        test,
        testoninputs,
        trainloop,
        forward,
        Args,
        StackValue,
        createinputstates,
        accuracyonexamples,
        trainbatch
end

val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

#########################
#     Conversion        #
#########################
function convert_discrete_to_continuous(
    discrete::DiscreteVMState,
    allvalues::Array = allvalues,
)::VMState
    stackdepth = discrete.stackdepth
    programlen = discrete.programlen
    inputlen = discrete.inputlen
    outputlen = discrete.outputlen

    contstate = VMState(stackdepth, programlen, allvalues, inputlen, outputlen)
    cont_instrpointer =
        onehot(discrete.instrpointer, StackValue.([i for i = 1:programlen])) * 1.0f0

    discretestack = Array{Any,1}(undef, stackdepth)
    fill!(discretestack, StackValue())
    for (i, x) in enumerate(discrete.stack)
        discretestack[i] = x
    end
    contstack = onehotbatch(discretestack, allvalues) * 1.0f0

    discreteinput = Array{Any,1}(undef, inputlen)
    fill!(discreteinput, StackValue())
    for (i, x) in enumerate(discrete.input)
        discreteinput[i] = x
    end
    continput = onehotbatch(discreteinput, allvalues) * 1.0f0

    discreteoutput = Array{Any,1}(undef, outputlen)
    fill!(discreteoutput, StackValue())
    for (i, x) in enumerate(discrete.output)
        discreteoutput[i] = x
    end
    contoutput = onehotbatch(discreteoutput, allvalues) * 1.0f0

    discretevariables = Array{Any,1}(undef, length(allvalues))
    fill!(discretevariables, StackValue())
    for (k, v) in discrete.variables
        discretevariables[findfirst(x -> x == k, allvalues)] = v
    end
    contvariables = onehotbatch(discretevariables, allvalues) * 1.0f0

    contishalted = onehot(discrete.ishalted, [false, true]) * 1.0f0
    VMState(
        cont_instrpointer |> device,
        contstate.stackpointer |> device,
        contstate.inputpointer |> device,
        contstate.outputpointer |> device,
        continput |> device,
        contoutput |> device,
        contstack |> device,
        contvariables |> device,
        contishalted |> device,
    )
end

function convert_continuous_to_discrete(
    contstate::VMState,
    allvalues = allvalues,
)::DiscreteVMState
    instrpointer = onecold(contstate.instrpointer)
    stackpointer = onecold(contstate.stackpointer)
    inputpointer = onecold(contstate.inputpointer)
    outputpointer = onecold(contstate.outputpointer)
    ishalted = onecold(contstate.ishalted, ishaltedvalues)
    stack = [allvalues[i] for i in onecold(contstate.stack)]
    input = [allvalues[i] for i in onecold(contstate.input)]
    output = [allvalues[i] for i in onecold(contstate.output)]
    variables = [allvalues[i] for i in onecold(contstate.variables)]
    #variables = onecold(contstate.stack)

    stack = circshift(stack, 1 - stackpointer) # Check if this actually makes sense with circshift
    input = circshift(input, 1 - inputpointer)
    output = circshift(output, 1 - outputpointer)
    # Dealing with blanks is tricky. It's not clear what is correct semantically
    newstack = CircularBuffer{StackValue}(size(contstate.stack)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in stack
        if x == StackValue()
            push!(newstack, StackValue())
        else
            # TODO convert to int ?
            push!(newstack, x)
        end
    end

    newinput = CircularBuffer{StackValue}(size(contstate.input)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in input
        if x == StackValue()
            push!(newinput, StackValue())
        else
            push!(newinput, x)
        end
    end

    newoutput = CircularBuffer{StackValue}(size(contstate.output)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in output
        if x == StackValue()
            push!(newoutput, StackValue())
        else
            push!(newoutput, x)
        end
    end

    #TODO Also push! is wrong! Need to add test that actually tests this (random init input, output, etc)
    newvariables = DefaultDict{StackValue,StackValue}(StackValue())
    # default blank? blank isn't technically in it
    # Use missing instead of blank?
    for (i, x) in enumerate(variables)
        newvariables[allvalues[i]] = x
    end
    DiscreteVMState(
        instrpointer = instrpointer,
        input = newinput,
        output = newoutput,
        stack = newstack,
        variables = newvariables,
        ishalted = ishalted,
    )
end

function ==(x::CircularDeque, y::CircularDeque)
    x.capacity != y.capacity && return false
    length(x) != length(y) && return false
    for (i, j) in zip(x, y)
        i == j || return false
    end
    true
end

# function ==(x::CircularBuffer, y::CircularBuffer)
#     x.capacity != y.capacity && return false
#     length(x) != length(y) && return false
#     for (i, j) in zip(x, y)
#         i == j || return false
#     end
#     true
# end

function ==(x::DiscreteVMState, y::DiscreteVMState)
    x.instrpointer == y.instrpointer &&
        x.stack == y.stack &&
        x.variables == y.variables &&
        x.ishalted == y.ishalted
end

function ==(x::VMState, y::VMState)
    x.instrpointer == y.instrpointer &&
        x.stackpointer == y.stackpointer &&
        x.stack == y.stack
end

###############################
# Initialization functions
###############################

function create_random_discrete_program(len, instructions)
    program = [rand(instructions) for i = 1:len]
end

function create_trainable_mask(programlen, inputlen)
    mask = falses(programlen)
    mask[inputlen+1:end] .= true
    mask
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
    variablemaskeds = Array{StackFloatType}(undef, (size(variablemasked)..., numexamples))
    for i = 1:numexamples
        newvariablemasked = copy(variablemasked)
        for col in eachcol(newvariablemasked)
            shuffle!(col)
        end
        variablemaskeds[:, :, i] = newvariablemasked  # Batch
    end
    variablemaskeds
end

function runprogram(state::VMState, program::Array, instructions::Vector{Function}, maxticks::Int)
    for i = 1:maxticks
        state = super_step(state, program, instructions)
    end
    state
end

function runprogram(state::DiscreteVMState, program::Array{Function}, maxticks::Int)
    for i in 1:maxticks
        runnextinstr(state, program)
        if state.ishalted
            break
        end
    end
    state
end

function runnextinstr(state::DiscreteVMState, program)
    instr = program[state.instrpointer]
    instr(state)
end

function loss(ŷ::VMState, y::VMState)
    ŷoutput = ŷ.output .* permutedims(ŷ.outputpointer)
    youtput = y.output .* permutedims(y.outputpointer)
    crossentropy(ŷoutput, youtput)
end

function accuracy(hidden, target, trainmask)
    samemax = onecold(hidden) .== onecold(target)
    result = (sum(samemax) - sum(1 .- trainmask)) / sum(trainmask)
end

function accuracyonexamples(hidden::Matrix{T}, target::Matrix{T}, instructions, examples, maxticks) where T <: Number
    # TODO does this need softmaxmask?
    predprogram = instructions[onecold(hidden)]
    targetprogram = instructions[onecold(target)]
    correctexamples = []
    for example in examples
        targetexample = deepcopy(example)
        predexample = deepcopy(example)
        runprogram(targetexample, targetprogram, maxticks)
        runprogram(predexample, predprogram, maxticks)
        push!(correctexamples, predexample.output == targetexample.output)
    end
    sum(correctexamples) / length(examples)
end

function approxoutputaccuracy(hidden::Matrix{Number}, target::Matrix{Number}, instructions::Vector{Function}, examples, maxticks)
    discretepredprogram = instructions[onecold(hidden)]
    discretetargetprogram = instructions[onecold(target)]
    correctexamples = []
    for example in examples
        targetexample = deepcopy(example)
        predexample = deepcopy(example)
        runprogram(targetexample, discretetargetprogram, maxticks)
        runprogram(predexample, discretepredprogram, maxticks)
        push!(correctexamples, sum(predexample.output .== targetexample.output))
        #TODO plot histogram? Show best example?
        @info "Approx Output Accuracy ---- input: $(example.input) target: $(targetexample.output) pred: $(predexample.output)"
    end
    sum(correctexamples) / length(examples)
end

function test(
    hiddenprogram,
    targetprogram,
    startstate,
    instructions,
    maxticks,
    trainmaskfull,
)
    program = softmaxmask(trainmaskfull, hiddenprogram)
    target = runprogram(startstate, targetprogram, instructions, maxticks)
    prediction = runprogram(startstate, program, instructions, maxticks)
    loss(prediction, target)
end

function testoninputs(
    hiddenprogram,
    inputstates,
    targetstates,
    instructions,
    maxticks,
    trainmaskfull,
)
    loss = 0
    for (i, startstate) in enumerate(inputstates)
        loss += forward(
            startstate,
            targetstates[i], #TODO awkward
            instructions,
            maxticks,
            hiddenprogram,
            trainmaskfull,
        )
    end
    return loss / length(inputstates)
end

function forward(state, target, instructions, maxticks, hiddenprogram, trainmaskfull)
    program = softmaxmask(trainmaskfull, hiddenprogram)
    pred = runprogram(state, program, instructions, maxticks)
    loss(pred, target)
end

function trainsingle(
    hiddenprogram,
    startstate,
    target,
    instructions,
    programlen,
    trainmaskfull;
    numexamples = 4,
    opt = Descent(0.1),
)
    @showprogress for i = 1:numexamples
        grads = gradient(
            forward,
            startstate,
            target,
            instructions,
            programlen,
            hiddenprogram,
            trainmaskfull,
        )[end-1] # end-1 for hidden?
        grads = grads .* trainmaskfull
        Optimise.update!(opt, hiddenprogram, grads)
    end
end

function createinputstates(state; num = 10)
    inputs = [rand(1:size(state.stack)[1], length(state.inputpointer)) for i = 1:num]
    inputstates = []

    for input in inputs
        state = VMState(
            state.instrpointer,
            state.stackpointer,
            state.inputpointer,
            state.outputpointer,
            fillinput(input, length(state.inputpointer)),
            state.output,
            state.stack,
            state.variables,
            state.ishalted,
        )
        push!(inputstates, state)
    end
    inputstates
end

function trainbatch(
    hiddenprogram,
    instructions,
    maxticks,
    inputstates,
    targetstates,
    trainmaskfull;
    batchsize = 4,
    epochs = 5,
    opt = Descent(0.1),
)
    testlength = min(length(inputstates), 50)
    grads = similar(hiddenprogram)
    grads .= 0
    for epoch in 1:epochs
        progressbar = Progress(length(inputstates))
        Threads.@threads for (i, startstate) in collect(enumerate(inputstates))
            grads = grads .+ gradient(
                forward,
                startstate,
                targetstates[i], #TODO awkward
                instructions,
                maxticks,
                hiddenprogram,
                trainmaskfull,
            )[end-1]
            grads = grads .* trainmaskfull
            if i % batchsize == 0 && i != 0
                Optimise.update!(opt, hiddenprogram, grads)
                grads .= 0
            end
            next!(progressbar)
        end
        loss = testoninputs(
            hiddenprogram,
            inputstates[1:testlength],
            targetstates[1:testlength],
            instructions,
            maxticks,
            trainmaskfull,
        )
        @info "epoch: $(epoch)/$(epochs) loss: $(loss)"
    end
end
end
