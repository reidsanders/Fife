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
    blankstack

using FifeTypes

import Fife: instr_pushval!, args

using Parameters: @with_kw
using Flux
using Flux: onehot, onehotbatch, glorot_uniform, gradient

#set inputlen
args.inputlen = 0

instr_pushval!(val::StackValue, state::VMState) = instr_pushval!(val, state, allvalues)
val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

instructions = [
    [
        instr_pass!,
        instr_halt!,
        # instr_pushval!,
        # instr_pop!,
        instr_dup!,
        instr_swap!,
        instr_add!,
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
    val_instructions
]


num_instructions = length(instructions)

discrete_program = create_random_discrete_program(args.programlen, instructions)
#TODO add a write! at end? or scattered throughout
target_program = convert(Array{StackFloatType}, onehotbatch(discrete_program, instructions))
trainmask = create_trainable_mask(args.programlen, args.inputlen)
hiddenprogram = deepcopy(target_program)
hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))


# Initialize
trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device

hiddenprogram = hiddenprogram |> device
program = softmaxmask(hiddenprogram, trainmaskfull) |> device
target_program = target_program |> device
hiddenprogram = hiddenprogram |> device
trainmask = trainmask |> device

blank_state =
    VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
blank_state2 =
    VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)

check_state_asserts(blank_state)

# TODO need to generate dataset of input, target
target = runprogram(blank_state, target_program, instructions, args.max_ticks)

# gradprogpart = partial(
#     gradient,
#     forward,
#     blank_state,
#     target,
#     instructions,
#     args.programlen,
#     trainmaskfull,
# )

first_program = deepcopy(program)
# opt = ADAM(0.002) 
opt = Descent(0.001)

######################################
# runprogram program train
######################################
first_loss = test(
    hiddenprogram,
    target_program,
    blank_state,
    instructions,
    args.programlen,
    trainmaskfull,
)
first_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)

@time trainloopsingle(
    hiddenprogram,
    blank_state,
    target,
    instructions,
    args.programlen,
    trainmaskfull,
    numexamples = 10000,
)

second_loss = test(
    hiddenprogram,
    target_program,
    blank_state,
    instructions,
    args.programlen,
    trainmaskfull,
)
second_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
