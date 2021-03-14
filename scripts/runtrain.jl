# using Pkg
# Pkg.activate(".")
using Fife
using Fife: 
    valhot,
    pushtostack,
    popfromstack,
    op_probvec,
    normalize_stackpointer,
    create_random_discrete_program,
    create_trainable_mask,
    super_step,
    softmaxmask,
    applyfullmask
using Parameters: @with_kw
using Flux
using Flux:
    onehot,
    onehotbatch,
    glorot_uniform,
    gradient

######################################
# Global initialization
######################################

# @with_kw mutable struct TrainArgs
#     batchsize::Int = 8
#     lr::Float32 = 2e-4
#     epochs::Int = 50
#     stackdepth::Int = 100
#     programlen::Int = 50
#     inputlen::Int = 20 # frozen part, assumed at front for now
#     max_ticks::Int = 40
#     maxint::Int = 50
#     trainsetsize::Int = 32
#     usegpu::Bool = false
# end
# args = TrainArgs()

@with_kw mutable struct Args
    batchsize::Int = 2
    lr::Float32 = 2e-4
    epochs::Int = 2
    stackdepth::Int = 10
    programlen::Int = 10
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 20
    trainsetsize::Int = 10
    usegpu::Bool = false
    StackFloatType::Type = Float32
    StackValueType::Type = Int

    ## TODO initialization function inside Args / or inside Fife module (then export inside args?)
end
args = Args()

device,
largevalue,
coercetostackvaluepart,
numericvalues,
nonnumericvalues,
allvalues,
ishaltedvalues,
blanks,
blankstack = create_dependent_values(args)

val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

#instructions = [[instr_gotoif!, instr_dup!]; val_instructions]
#instructions = [[instr_pass!, instr_dup!]; val_instructions]
#instructions = [[instr_pass!]; val_instructions]

instructions = [
    [
        instr_pass!,
        instr_halt!,
        # instr_pushval!,
        # instr_pop!,
        instr_dup!,
        instr_swap!,
        instr_add!,
        instr_sub!,
        instr_mult!,
        instr_div!,
        instr_not!,
        instr_and!,
        # instr_goto!,
        instr_gotoif!,
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

discrete_programs = [
    [
        create_random_discrete_program(args.inputlen, instructions)
        discrete_program[end-args.inputlen:end]
    ] for x = 1:args.trainsetsize
]

target_program = convert(Array{Float32}, onehotbatch(discrete_program, instructions))
trainmask = create_trainable_mask(args.programlen, args.inputlen)
hiddenprogram = deepcopy(target_program)
hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))


# Initialize

trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device
applyfullmaskprog = partial(applyfullmask, trainmaskfull)
applyfullmasktohidden = partial((mask, prog) -> mask .* prog, trainmaskfull)

hiddenprogram = hiddenprogram |> device
program = softmaxmask(hiddenprogram, trainmaskfull) |> device
target_program = target_program |> device
hiddenprogram = hiddenprogram |> device
trainmask = trainmask |> device


blank_state = VMState(args.stackdepth, args.programlen, allvalues)
blank_state2 = VMState(args.stackdepth, args.programlen, allvalues)

check_state_asserts(blank_state)

# TODO need to generate dataset of input, target
target = runprogram(blank_state, target_program, instructions, 10)
# target = runprogram(blank_state, target_program, instructions, args.programlen)

gradprogpart =
    partial(gradient, forward, blank_state, target, instructions, args.programlen, trainmaskfull) # Partial?

first_program = deepcopy(program)
# opt = ADAM(0.002) 
opt = Descent(0.1)

######################################
# runprogram program train
######################################
first_loss = test(hiddenprogram, target_program, blank_state, instructions, args.programlen, trainmaskfull)
first_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)

@time trainloopsingle(hiddenprogram, numexamples = 3)

second_loss =
    test(hiddenprogram, target_program, blank_state, instructions, args.programlen, trainmaskfull)
second_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
