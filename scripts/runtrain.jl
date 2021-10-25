# using Pkg
# Pkg.activate(".")
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
    device,
    numericvalues,
    nonnumericvalues,
    allvalues,
    ishaltedvalues,
    blanks,
    blankstack,
    trainsingle,
    trainbatch,
    createinputstates

using ParameterSchedulers
using ParameterSchedulers: Scheduler


import Fife: instr_pushval!, args, show
using Parameters: @with_kw
using Flux
using Flux: onehot, onehotbatch, glorot_uniform, gradient, onecold, hidden
using Random
Random.seed!(123);

args.programlen = 5
args.maxticks = 10
args.lr = .05
opt = Scheduler(Cos(λ0 = args.lr, λ1 = args.lr * 1e2, period = 10), Momentum())

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
#####

targetprogram = convert(Array{args.StackFloatType}, onehotbatch(discrete_program, instructions))
trainmask = create_trainable_mask(args.programlen, 0)
hiddenprogram = deepcopy(targetprogram)
hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))

#TODO run multiple inputs
#TODO define some basic programs to try instead of randomly

# Initialize
trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device

hiddenprogram = hiddenprogram |> device
program = softmaxmask(trainmaskfull, hiddenprogram) |> device
targetprogram = targetprogram |> device
hiddenprogram = hiddenprogram |> device
trainmask = trainmask |> device
state =
    VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)

@info "Create inputstates"
inputstates = createinputstates(state, num = 100)
targetstates = [runprogram(input, targetprogram, instructions, args.maxticks) for input in inputstates]
discreteinputstates = [convert_continuous_to_discrete(state) for state in inputstates]

first_program = deepcopy(program)

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
first_exampleaccuracy = accuracyonexamples(hiddenprogram, targetprogram, instructions, discreteinputstates, args.maxticks)
@time trainbatch(
    hiddenprogram,
    instructions,
    args.maxticks,
    inputstates,
    targetstates,
    trainmaskfull,
    batchsize = 50,
    epochs = 8,
    opt = opt
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
second_exampleaccuracy = accuracyonexamples(hiddenprogram, targetprogram, instructions, discreteinputstates, args.maxticks)
approx_accuracy = approxoutputaccuracy(hiddenprogram, targetprogram, instructions, discreteinputstates[1:10], args.maxticks)
predprogram = instructions[onecold(hiddenprogram)]
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
@show first_exampleaccuracy
@show second_exampleaccuracy