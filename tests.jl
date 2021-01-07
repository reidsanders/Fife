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
    instructionpointer = zeros(Float32, programlen, )
    stackpointer = zeros(Float32, stackdepth, )
    #instructionpointer = rand(Float32, programlen, )
    #stackpointer = rand(Float32, stackdepth, )
    stack = normit(stack)
    instructionpointer[1] = 1.f0
    stackpointer[1] = 1.f0
    state = VMState(
        instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
    )
    #normit(state)
end

allvalues = [i for i in 0:5]
instr_2 = partial(instr_val, valhot(2, allvalues))

blank_state = init_state(3, 4, allvalues)
blank_state_random = init_random_state(3, 4, allvalues)
check_state_asserts(blank_state)
check_state_asserts(blank_state_random)

newstate = instr_dup(blank_state_random)
newstate_instr2 = instr_2(blank_state_random)

check_state_asserts(newstate)
check_state_asserts(newstate_instr2)