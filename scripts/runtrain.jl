using Fife
using Fife:
    valhot,
    pushtostack,
    popfromstack,
    op_probvec,
    normalize_stackpointer,
    normalize_iopointers,
    create_random_discrete_program,
    create_trainable_mask,
    super_step,
    softmaxmask,
    applyfullmask,
    allvalues,
    targetdevice,
    numericvalues,
    nonnumericvalues,
    allvalues,
    ishaltedvalues,
    blanks,
    blankstack,
    trainsingle,
    trainbatch!,
    createinputstates

using ParameterSchedulers
using ParameterSchedulers: Scheduler
using TensorBoardLogger
using Logging

import Fife: instr_pushval!, args
using Parameters: @with_kw
using Flux
using Flux: onehot, onehotbatch, glorot_uniform, glorot_normal, gradient, onecold, hidden
using Random
using Dates
Random.seed!(123);
import Base: show

"""
Show
"""
function show(io::IO, x::StackValue)
    if x.blank
        print(io, "∅")
    elseif x.max
        print(io, "^")
    elseif x.min
        print(io, "v")
    else
        print(io, x.val)
    end
end


x = [StackValue(), StackValue(100)]
@info " StackValue: $(x)"

# args.programlen = 5
args.experimentname = "regular-train"
args.trainsize = 32
args.maxticks = 10
args.epochs = 1
args.lr = 0.1
opt = ADAM(args.lr)
# opt = Scheduler(TriangleExp(λ0 = args.lr, λ1 = args.lr * 20, period = 10, γ = .95), Momentum())
# opt = Scheduler(Cos(λ0 = args.lr, λ1 = args.lr * 1e2, period = 10), Momentum())

instr_pushval!(val::StackValue, state::VMState) = instr_pushval!(val, state, allvalues)
val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

instructions = [
    instr_pass!,
    instr_halt!,
    # instr_pushval!,
    instr_pop!,
    instr_dup!,
    instr_swap!,
    instr_add!,
    instr_read!,
    instr_write!,
    instr_sub!,
    instr_mult!,
    instr_div!,
    instr_not!,
    instr_and!,
    instr_goto!,
    instr_gotoif!,
    # instr_iseq!,
    # instr_isgt!,
    # instr_isge!,
    # instr_store!,
    # instr_load!
]


@info "Setting up instructions and training data"
num_instructions = length(instructions)


##### define program to learn
# discrete_program = create_random_discrete_program(args.programlen, instructions)
# discrete_program[1:2] .= instr_read!
# discrete_program[end-1:end] .= instr_write!

discrete_program = [instr_read!, instr_read!, instr_swap!, instr_write!, instr_write!]
# discrete_program = [instr_read!, instr_write!]
# discrete_program = [instr_read!, instr_write!, instr_write!]
args.programlen = length(discrete_program)
#####

targetprogram =
    convert(Array{args.StackFloatType}, onehotbatch(discrete_program, instructions))
trainmask = create_trainable_mask(args.programlen, 0)
hiddenprogram = deepcopy(targetprogram)
# hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))
# hiddenprograms = [glorot_normal(size(hiddenprogram)) for i = 1:512]
hiddenprogram .= 0
# hiddenprogram[:, trainmask] .= 0

# Initialize
trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> targetdevice

hiddenprogram = hiddenprogram |> targetdevice
program = softmaxmask(trainmaskfull, hiddenprogram) |> targetdevice
targetprogram = targetprogram |> targetdevice
hiddenprogram = hiddenprogram |> targetdevice
trainmask = trainmask |> targetdevice
state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)

@info "Create inputstates"
inputstates = createinputstates(state, num = args.trainsize)
targetstates =
    [runprogram(input, targetprogram, instructions, args.maxticks) for input in inputstates]
discreteinputstates = [convert_continuous_to_discrete(state) for state in inputstates]

first_program = deepcopy(program)

#create tensorboard logger
logdir = "logs/runs/$(args.experimentname)_$(Dates.now())"
logger = TBLogger(logdir, tb_append)
with_logger(logger) do
    @info "hyperparams" batchsize=args.batchsize lr=args.lr epochs=args.epochs stackdepth=args.stackdepth programlen=args.programlen inputlen=args.inputlen outputlen=args.outputlen maxticks=args.maxticks maxint=args.maxint trainsize=args.trainsize usegpu=args.usegpu
end

#function to log information after every epoch
# relative time, separate train and test loss. approx acc. epoch. lr, all args hyperparams. 
function TBCallback(; params = Dict())
    testlength = min(length(inputstates), 32)
    with_logger(logger) do
        trainexampleaccuracy = accuracyonexamples(
            hiddenprogram,
            targetprogram,
            instructions,
            discreteinputstates[1:testlength],
            args.maxticks,
        )
        testloss = testoninputs(
            hiddenprogram,
            inputstates[1:testlength],
            targetstates[1:testlength],
            instructions,
            args.maxticks,
            trainmaskfull,
        )
        programprobs = hiddenprogram |> softmax
        try 
            @info "train" loss=params["loss"] accuracy=trainexampleaccuracy step=params["step"] steptime=params["time"] lr=opt.eta programprobs = programprobs
        catch e
            if isa(e, KeyError)
                @warn e
            end
        end
        @info "test" loss=testloss step=step lr=opt.eta
    end
end

######################################
# runprogram program train
######################################
@info "Get first loss and accuracy"
first_loss = testoninputs(
    hiddenprogram,
    inputstates,
    targetstates,
    instructions,
    args.maxticks,
    trainmaskfull,
)
@info "first loss: $(first_loss)"
first_accuracy = accuracy(hiddenprogram |> cpu, targetprogram |> cpu, trainmask |> cpu)
first_exampleaccuracy = accuracyonexamples(
    hiddenprogram,
    targetprogram,
    instructions,
    discreteinputstates,
    args.maxticks,
)

@time trainbatch!(
    hiddenprogram,
    instructions,
    args.maxticks,
    inputstates,
    targetstates,
    trainmaskfull,
    batchsize = args.batchsize,
    epochs = args.epochs,
    opt = opt,
    cb = Flux.throttle(TBCallback, 30),
)

second_loss = testoninputs(
    hiddenprogram,
    inputstates,
    targetstates,
    instructions,
    args.maxticks,
    trainmaskfull,
)
second_accuracy = accuracy(hiddenprogram |> cpu, targetprogram |> cpu, trainmask |> cpu)
second_exampleaccuracy = accuracyonexamples(
    hiddenprogram,
    targetprogram,
    instructions,
    discreteinputstates,
    args.maxticks,
)
approx_accuracy = approxoutputaccuracy(
    hiddenprogram,
    targetprogram,
    instructions,
    discreteinputstates[1:10],
    args.maxticks,
)
predprogram = instructions[onecold(hiddenprogram)]
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
@show first_exampleaccuracy
@show second_exampleaccuracy