include("main.jl")

######################################
# Global initialization
######################################

#=
@with_kw mutable struct Args
    batchsize::Int = 32
    lr::Float32 = 2e-4
    epochs::Int = 50
    stackdepth::Int = 100
    programlen::Int = 50
    inputlen::Int = 20 # frozen part, assumed at front for now
    max_ticks::Int = 40
    maxint::Int = 50
    usegpu::Bool = false
end

args = Args()

# use_cuda = false
if args.usegpu
    global device = gpu
    # @info "Training on GPU"
else
    global device = cpu
    # @info "Training on CPU"
end

numericvalues = [i for i in 0:args.maxint]
nonnumericvalues = ["blank"]
allvalues = [nonnumericvalues; numericvalues]
=#


instr_gotoif! = partial(instr_gotoiffull!, valhot(0, allvalues), nonnumericvalues)

val_instructions = [partial(instr_pushval!, i) for i in numericvalues]
instructions = [[instr_gotoif!, instr_dup!]; val_instructions]
num_instructions = length(instructions)

discrete_program = create_random_discrete_program(args.programlen, instructions)
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


blank_state = init_state(args.stackdepth, args.programlen, allvalues)
blank_state2 = init_state(args.stackdepth, args.programlen, allvalues)

check_state_asserts(blank_state)

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

@time trainloopsingle(hiddenprogram, numexamples = 10)

second_loss =
    test(hiddenprogram, target_program, blank_state, instructions, args.programlen)
second_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy
