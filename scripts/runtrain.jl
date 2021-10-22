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

# include("../src/types.jl")
# using .FifeTypes

import Fife: instr_pushval!, args

using Parameters: @with_kw
using Flux
using Flux: onehot, onehotbatch, glorot_uniform, gradient

args.programlen = 5
args.maxticks = 10
args.lr = .1

instr_pushval!(val::StackValue, state::VMState) = instr_pushval!(val, state, allvalues)
val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

instructions = [
    instr_pass!,
    instr_halt!,
    # instr_pushval!,
    # instr_pop!,
    instr_dup!,
    instr_swap!,
    instr_add!,
    instr_read!,
    instr_write!,
    # instr_sub!,
    # instr_mult!,
    # instr_div!,
    # instr_not!,
    # instr_and!,
    # instr_goto!,
    # instr_gotoif!,
    # instr_iseq!,
    # instr_isgt!,
    # instr_isge!,
    # instr_store!,
    # instr_load!
]



num_instructions = length(instructions)

# discrete_program = create_random_discrete_program(args.programlen, instructions)
# discrete_program[end] = instr_write!
# discrete_program[1] = instr_read!

discrete_program = [instr_read!, instr_read!, instr_swap!, instr_write!, instr_write!]
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

startstate = VMState(
    state.instrpointer,
    state.stackpointer,
    state.inputpointer,
    state.outputpointer,
    fillinput([2, 5, 3], args.inputlen),
    state.output,
    state.stack,
    state.variables,
    state.ishalted,
)
discretestartstate = convert_continuous_to_discrete(startstate)

inputstates = createinputstates(startstate, num = 100)
targetstates = [runprogram(input, targetprogram, instructions, args.maxticks) for input in inputstates]
# datastates = [(inputstate, targetstate) for (i,)]
discreteinputstates = [convert_continuous_to_discrete(state) for state in inputstates]

check_state_asserts(startstate)

first_program = deepcopy(program)

######################################
# runprogram program train
######################################
first_loss = test(
    hiddenprogram,
    targetprogram,
    startstate,
    instructions,
    args.programlen,
    trainmaskfull,
)
first_accuracy = accuracy(hiddenprogram |> cpu, targetprogram |> cpu, trainmask |> cpu)
first_exampleaccuracy = accuracyonexamples(hiddenprogram, targetprogram, instructions, discreteinputstates, args.maxticks)

@time trainbatch(
    hiddenprogram,
    instructions,
    args.programlen,
    args.maxticks,
    inputstates,
    targetstates,
    trainmaskfull,
    batchsize = 10,
    epochs = 10,
    opt = ADAM(args.lr)
)

second_loss = test(
    hiddenprogram,
    targetprogram,
    startstate,
    instructions,
    args.maxticks,
    trainmaskfull,
)
second_accuracy = accuracy(hiddenprogram |> cpu, targetprogram |> cpu, trainmask |> cpu)
second_exampleaccuracy = accuracyonexamples(hiddenprogram, targetprogram, instructions, discreteinputstates, args.maxticks)
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
@show first_exampleaccuracy
@show second_exampleaccuracy