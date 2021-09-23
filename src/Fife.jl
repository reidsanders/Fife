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

#TODO remove mutable / make const?
@with_kw mutable struct Args
    batchsize::Int = 2
    lr::Float32 = 2e-4
    epochs::Int = 2
    stackdepth::Int = 11
    programlen::Int = 13
    inputlen::Int = 7
    outputlen::Int = 8
    max_ticks::Int = 5
    maxint::Int = 20
    trainsetsize::Int = 10
    usegpu::Bool = false
    StackFloatType::Type = Float32
    StackValueType::Type = Int

    ## TODO initialization function inside Args / or inside Fife module (then export inside args?)
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
    largevalue = floor(args.StackValueType, sqrt(typemax(args.StackValueType)))
    coercetostackvaluepart = partial(
        coercetostackvalue,
        args.StackValueType,
        -args.maxint,
        args.maxint,
        largevalue,
    )
    numericvalues = [[-largevalue]; [i for i = -args.maxint:args.maxint]; [largevalue]]
    nonnumericvalues = ["blank"]
    allvalues = [nonnumericvalues; numericvalues]
    ishaltedvalues = [false, true]
    blanks = fill("blank", args.stackdepth)
    blankstack = onehotbatch(blanks, allvalues)
    inputblanks = fill("blank", args.inputlen)
    blankinput = onehotbatch(inputblanks, allvalues)
    outputblanks = fill("blank", args.outputlen)
    blankoutput = onehotbatch(outputblanks, allvalues)

    (
        device,
        largevalue,
        coercetostackvaluepart,
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
largevalue,
coercetostackvaluepart,
numericvalues,
nonnumericvalues,
allvalues,
ishaltedvalues,
blanks,
blankstack, 
blankinput,
blankoutput = create_dependent_values(args)

include("discreteinterpreter.jl")
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
        trainloop,
        trainloopsingle,
        trainbatch!,
        forward,
        Args
end

#########################
#     StackValueType    #
#########################
# @with_kw struct StackValue
#     val::Int = 0
#     blank::Bool = true
#     max::Bool = false
#     min::Bool = false
#     # OrderedPair(x1,x2,x3,x4) = x > y ? error("out of order") : new(x,y)
# end
# # StackValue(x) = StackValue(val = x)
# function StackValue(x)
#     if x >= args.maxint
#         return StackValue(val=0,blank=false,max=true,min=false)
#     elseif x <= -args.maxint
#         return StackValue(val=0,blank=false,max=false,min=true)
#     else
#         return StackValue(val=x,blank=false,max=false,min=false)
#     end
# end

# function +(x::StackValue, y::StackValue)
#     if x.blank || y.blank
#         return StackValue()
#     elseif x.max
#         if y.min 
#             return StackValue(0)
#         else
#             return StackValue(blank=false, max=true)
#         end
#     elseif y.max
#         if x.min 
#             return StackValue(0)
#         else
#             return StackValue(blank=false, max=true)
#         end
#     elseif y.min || x.min
#         return StackValue(blank=false, min=true)
#     end

#     StackValue(x.val + y.val)
# end

# function *(x::StackValue, y::StackValue)
#     if x.blank || y.blank
#         return StackValue()
#     elseif x.max & y.min || x.min & y.max
#         return StackValue(blank=false, min=true)
#     elseif x.max & y.max || x.min & y.min
#         return StackValue(blank=false, max=true)
#     elseif x.max || y.max
#         return StackValue(blank=false, max=true)
#     elseif x.min || y.min
#         return StackValue(blank=false, min=true)
#     end

#     StackValue(x.val * y.val)
# end

# function *(x::Number, y::StackValue)
#     if y.blank
#         return StackValue()
#     elseif y.max & x > 0 || y.min & x < 0
#         return StackValue(blank=false, max=true)
#     elseif y.max & x < 0 || y.min & x > 0
#         return StackValue(blank=false, min=true)
#     end

#     StackValue(x * y.val)
# end
# x::StackValue * y::Number = y * x

# x::Number + y::StackValue = StackValue(x) + y
# x::StackValue + y::Number = y + x
# x::StackValue - y::StackValue = x + -1 * y
# x::Number - y::StackValue = StackValue(x) - y
# x::StackValue - y::Number = x - StackValue(y)


# function ==(x::StackValue, y::StackValue)
#     if y.blank && x.blank || y.max && x.max || y.min && x.min
#         return true
#     end
#     x.val == y.val
# end

# x::StackValue == y::Number = x == StackValue(y)
# x::Number == y::StackValue = StackValue(x) == y

# function convert(::Type{StackValue}, x::Number)
#     StackValue(x)
# end
#########################

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
    cont_instrpointer = onehot(discrete.instrpointer, [i for i = 1:programlen]) * 1.0f0

    discretestack = Array{Any,1}(undef, stackdepth)
    fill!(discretestack, "blank")
    for (i, x) in enumerate(discrete.stack)
        discretestack[i] = x
    end
    contstack = onehotbatch(discretestack, allvalues) * 1.0f0

    discreteinput = Array{Any,1}(undef, inputlen)
    fill!(discreteinput, "blank")
    for (i, x) in enumerate(discrete.input)
        discretestack[i] = x
    end
    continput = onehotbatch(discreteinput, allvalues) * 1.0f0

    discreteoutput = Array{Any,1}(undef, outputlen)
    fill!(discreteoutput, "blank")
    for (i, x) in enumerate(discrete.output)
        discreteoutput[i] = x
    end
    contoutput = onehotbatch(discreteoutput, allvalues) * 1.0f0

    discretevariables = Array{Any,1}(undef, stackdepth)
    fill!(discretevariables, "blank")
    for (k, v) in discrete.variables
        discretevariables[k] = v
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
    newstack = CircularBuffer{StackValueType}(size(contstate.stack)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in stack
        if x == "blank"
            break
            #TODO BUG
            # discrete can't have blank values on stack, but removing them 
            # is confusing and may mess up behavior if the superinterpreter is 
            # depending on there being a blank there
            # TODO either break or 
            # allow "blank" values on discrete stack? 
            # That would complicate the discrete operations a lot.
            # Use Union{Int, nothing}  ? Where nothing represents blank?
            # Or use magic int
            # Or use special type for blank, and have a union. Then dispatch based on that. Requires redefining *,+, etc
            # Or define composite type with bool? for isblank ismax ismin. Which still requires defining math ops
            # should the buffers be "prefilled" or not?
        else
            # TODO convert to int ?
            push!(newstack, x)
        end
    end

    newinput = CircularBuffer{StackValueType}(size(contstate.input)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in input
        if x == "blank"
            break
        else
            push!(newinput, x)
        end
    end

    newoutput = CircularBuffer{StackValueType}(size(contstate.output)[2]) # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in output
        if x == "blank"
            break
        else
            push!(newoutput, x)
        end
    end

    newvariables = DefaultDict{StackValueType,StackValueType}(0)
    # default blank? blank isn't technically in it
    # Use missing instead of blank?
    for x in variables
        if x == "blank"
            continue
        else
            # TODO convert to int ?
            newvariables[]
            push!(newstack, x)
        end
    end
    DiscreteVMState(instrpointer = instrpointer, input = newinput, output = newoutput, stack = newstack, ishalted = ishalted)
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

function create_random_inputs(len, instructions)
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

function runprogram(state, program, instructions, ticks)
    for i = 1:ticks
        state = super_step(state, program, instructions)
        # assert_no_nans(state)
    end
    state
end

# function loss(ŷ, y)
#     # TODO top of stack super position actual stack. add tiny amount of instrpointer just because
#     # Technically current instruction doesn't really matter
#     # zygote doesn't like variables not being used in loss
#     crossentropy(ŷ.stackpointer, y.stackpointer) +
#     crossentropy(ŷ.instrpointer, y.instrpointer) +
#     crossentropy(ŷ.stack, y.stack) +
#     crossentropy(ŷ.variables, y.variables) +
#     crossentropy(ŷ.ishalted, y.ishalted)
# end

function loss(ŷ::VMState, y::VMState)
    # TODO top of stack super position actual stack. add tiny amount of instrpointer just because
    # Technically current instruction doesn't really matter
    # zygote doesn't like variables not being used in loss
    crossentropy(ŷ.output, y.output)
end


function accuracy(hidden, target, trainmask)
    samemax = onecold(hidden) .== onecold(target)
    result = (sum(samemax) - sum(1 .- trainmask)) / sum(trainmask)
end

function test(
    hiddenprogram,
    targetprogram,
    blank_state,
    instructions,
    programlen,
    trainmaskfull,
)
    program = softmaxmask(hiddenprogram, trainmaskfull)
    target = runprogram(blank_state, targetprogram, instructions, programlen)
    prediction = runprogram(blank_state, program, instructions, programlen)
    loss(prediction, target)
end

function forward(state, target, instructions, programlen, hiddenprogram, trainmaskfull)
    program = softmaxmask(hiddenprogram, trainmaskfull)
    pred = runprogram(state, program, instructions, programlen)
    loss(pred, target)
end

function trainloopsingle(
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
        # grads = grad(forward, startstate, target, instructions, programlen, hiddenprogram, trainmaskfull)
        grads = grads .* trainmaskfull
        Optimise.update!(opt, hiddenprogram, grads)
    end
end
end
