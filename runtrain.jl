include("fife.jl")

######################################
# Global initialization
######################################

@with_kw mutable struct TrainArgs
    batchsize::Int = 8
    lr::Float32 = 2e-4
    epochs::Int = 50
    stackdepth::Int = 100
    programlen::Int = 50
    inputlen::Int = 20 # frozen part, assumed at front for now
    max_ticks::Int = 40
    maxint::Int = 50
    trainsetsize::Int = 32
    usegpu::Bool = false
end
args = TrainArgs()
include("parameters.jl")


#instr_gotoif! = partial(instr_gotoiffull!, valhot(0, allvalues), nonnumericvalues)

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
softmaxprog = partial(softmaxmask, trainmaskfull |> device)
applyfullmaskprog = partial(applyfullmask, trainmaskfull)
applyfullmasktohidden = partial((mask, prog) -> mask .* prog, trainmaskfull)

hiddenprogram = hiddenprogram |> device
program = softmaxprog(hiddenprogram) |> device
target_program = target_program |> device
hiddenprogram = hiddenprogram |> device
trainmask = trainmask |> device


blank_state = VMState(args.stackdepth, args.programlen, allvalues)
blank_state2 = VMState(args.stackdepth, args.programlen, allvalues)

check_state_asserts(blank_state)

# TODO need to generate dataset of input, target
target = run(blank_state, target_program, instructions, args.programlen)

gradprogpart =
    partial(gradient, forward, blank_state, target, instructions, args.programlen) # Partial?

first_program = deepcopy(program)
# opt = ADAM(0.002) 
opt = Descent(0.1)

######################################
# Run program train
######################################
first_loss = test(hiddenprogram, target_program, blank_state, instructions, args.programlen)
first_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)

@time trainloopsingle(hiddenprogram, numexamples = 3)

second_loss =
    test(hiddenprogram, target_program, blank_state, instructions, args.programlen)
second_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
