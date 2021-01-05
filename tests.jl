using Test

include("main.jl")

######################################
# Global initialization
######################################

@with_kw mutable struct Args
    batchsize::Int = 2
    lr::Float32 = 2e-4
    epochs::Int = 2
    stackdepth::Int = 10
    programlen::Int = 5
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 3
    usegpu::Bool = false
end

args = Args()

# use_cuda = false
if args.usegpu
    global device = gpu
    @info "Training on GPU"
else
    global device = cpu
    @info "Training on CPU"
end

function init_random_state(stackdepth, programlen, allvalues)
    stack = rand(Float32, length(allvalues), stackdepth)
    current_instruction = rand(Float32, programlen, )
    top_of_stack = rand(Float32, stackdepth, )
    state = VMState(
        current_instruction |> device,
        top_of_stack |> device,
        stack |> device,
    )
    normit(state)
end

function test_instr_dup()
    intvalues = [i for i in 0:args.maxint]
    nonintvalues = ["blank"]
    allvalues = [nonintvalues; intvalues]

    instr_gotoifnotzero = partial(instr_gotoifnotzerofull, valhot(0, allvalues), nonintvalues)

    val_instructions = [partial(instr_val, valhot(i, allvalues)) for i in intvalues]
    instructions = [[instr_gotoifnotzero, instr_dup]; val_instructions]
    num_instructions = length(instructions)

    discrete_program = create_random_discrete_program(args.programlen, instructions)
    target_program = convert(Array{Float32}, onehotbatch(discrete_program, instructions))
    trainmask = create_trainable_mask(args.programlen, args.inputlen)
    hiddenprogram = deepcopy(target_program)
    hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))


    # Initialize

    trainmaskfull = repeat(trainmask', outer=(size(hiddenprogram)[1], 1)) |> device
    softmaxprog = partial(softmaxmask, trainmaskfull |> device)
    applyfullmaskprog = partial(applyfullmask, trainmaskfull)
    applyfullmasktohidden = partial((mask, prog) -> mask .* prog, trainmaskfull)

    hiddenprogram = hiddenprogram |> device
    program = softmaxprog(hiddenprogram) |> device
    target_program = target_program |> device
    hiddenprogram = hiddenprogram |> device
    trainmask = trainmask |> device


    blank_state = init_state(args.stackdepth, args.programlen, allvalues)
    blank_state_random = init_random_state(args.stackdepth, args.programlen, allvalues)

    instr_dup(blank_state)
end


blank_state = init_state(10, 12, [i for i in 0:10])
blank_state_random = init_random_state(10, 12, [i for i in 0:10])
check_state_asserts(blank_state)
check_state_asserts(blank_state_random)

#new_state = test_instr_dup()