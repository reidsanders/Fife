using Pkg
Pkg.activate(".")
using Flux
using Flux: onehot, onehotbatch, onecold, crossentropy, logitcrossentropy, glorot_uniform, mse, epseltype
using Flux: Optimise
using CUDA
using Zygote
using Random
using LoopVectorization
using Debugger
using Base
import Base: +,-,*,length
using StructArrays
using BenchmarkTools
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using Profile
Random.seed!(123);

CUDA.allowscalar(false)

function partial(f, a...)
    ( (b...) -> f(a..., b...) )
end

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

struct VMState
    current_instruction::Union{Array{Float32},CuArray{Float32}}
    top_of_stack::Union{Array{Float32},CuArray{Float32}}
    stack::Union{Array{Float32},CuArray{Float32}}
end

struct VMSuperStates
    current_instructions::Union{Array{Float32},CuArray{Float32}}
    top_of_stacks::Union{Array{Float32},CuArray{Float32}}
    stacks::Union{Array{Float32},CuArray{Float32}} # num instructions x stack
end
# TODO just have an array of structs? maybe easier to make mutable, instructions can directly modify 


# Zygote.@adjoint VMSuperStates(x,y,z) = VMSuperStates(x,y,z), di -> (di.current_instructions, di.top_of_stacks, di.stacks)
# Zygote.@adjoint VMState(x,y,z) = VMState(x,y,z), di -> (di.current_instruction, di.top_of_stack, di.stack)
a::Number * b::VMState = VMState(a * b.current_instruction, a * b.top_of_stack, a * b.stack)
a::VMState * b::Number = b * a
a::VMState + b::VMState = VMState(a.current_instruction + b.current_instruction, a.top_of_stack + b.top_of_stack, a.stack + b.stack)
a::VMState - b::VMState = VMState(a.current_instruction - b.current_instruction, a.top_of_stack - b.top_of_stack, a.stack - b.stack)

length(a::VMSuperStates) = size(a.current_instructions)[3]
a::Union{Array,CuArray} * b::VMSuperStates = VMSuperStates(a .* b.current_instructions, a .* b.top_of_stacks, a .* b.stacks)
a::VMSuperStates * b::Union{Array,CuArray} = b * a

function super_step(state::VMState, program, instructions)
    # TODO instead of taking a state, take the separate arrays as args? Since CuArray doesn't like Structs
    # TODO batch the individual array (eg add superpose dimension -- can that be a struct or needs to be separate?)
    newstates = [instruction(state) for instruction in instructions]
    current_instructions = cat([x.current_instruction for x in newstates]..., dims=3)
    top_of_stacks = cat([x.top_of_stack for x in newstates]..., dims=3)

    stacks = cat([x.stack for x in newstates]..., dims=3)

    states = VMSuperStates(
        current_instructions,
        top_of_stacks,
        stacks,
    )
    current = program .* state.current_instruction'
    summed = sum(current, dims=2) 
    summed = reshape(summed, (1, 1, :))
    scaledstates = summed * states 
    reduced = VMState( 
        sum(scaledstates.current_instructions, dims=3)[:,:,1],
        sum(scaledstates.top_of_stacks, dims=3)[:,:,1],
        sum(scaledstates.stacks, dims=3)[:,:,1],
    )
    normit(reduced)
end

###############################
# Instructions
###############################

function instr_dup(state::VMState)
    #= 
    DUP Should duplicate top of stack, and push to top of stack =#
    new_top_of_stack = roll(state.top_of_stack, -1)
    new_stack = state.stack .* (1.f0 .- state.top_of_stack') .+ state.stack .* new_top_of_stack'
    new_current_instruction = roll(state.current_instruction, 1)

    VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
    
end

function instr_swap(state::VMState)
    #= 
    DUP Should duplicate top of stack, and push to top of stack =#
    new_top_of_stack = roll(state.top_of_stack, -1)
    new_stack = state.stack .* (1.f0 .- state.top_of_stack') .+ state.stack .* new_top_of_stack'
    new_current_instruction = roll(state.current_instruction, 1)

    VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
    
end


    
function instr_gotoifnotzerofull(zerovec, nonintvalues, state::VMState)
    #= 
    GOTO takes top two elements of stack. If top is not zero, goto second element (or end, if greater than program len)

    take stack. apply bitmask for 0 / mult by zero onehot (or select col by index, but slow on gpu)
    1.- , then mult as prob vector. 
    Apply top of stack prob? =#

    # TODO don't recalc 
    stackscaled = state.stack .* state.top_of_stack'
    probofgoto = 1 .- sum(stackscaled .* zerovec, dims=1)
    firstpoptop = roll(state.top_of_stack, 1)
    secondstackscaled = state.stack .* firstpoptop' 
    jumpvalprobs = sum(secondstackscaled, dims=2)

    # Assume ints start at 0 should skip 0?
    # Constraint -- length(intvalues) > length of current instruction ?
    jumpvalprobs = jumpvalprobs[length(nonintvalues) + 1:end]
    # TODO length(current_instruction) compare and truncate
    jumpvalprobs = jumpvalprobs[1:end]
    currentinstructionforward = (1.f0 - sum(jumpvalprobs)) * roll(state.current_instruction, 1)
    new_current_instruction = currentinstructionforward .+ jumpvalprobs[1:length(state.current_instruction)]
    newtop = roll(firstpoptop, 1)

    # jumpvalprobs[:end]
    # TODO set blank / zero to zero? zero goes to first? greater than length(program) goes to end?
    # for each value 0-max in stack, 
    # (??sum diagonal of equivalent jump locations. eg 0 at x is equivalent to 1 at x+1 on stack)
    # TODO if not int value, set to roll 1 forward?
    # TODO add jumpvalprobs to current_instruction? Then normalize?

    VMState(
        new_current_instruction,
        newtop,
        state.stack,
    )
    
end

# TODO normit after most instr (?)
# TODO def normit for  all zero case


instr_pass(state::VMState) = state


function instr_val(valhotvec, state::VMState)
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time
    # sizehint
    # set return type to force allocation
    # display(state.top_of_stack)
    new_top_of_stack = roll(state.top_of_stack, -1)
    new_current_instruction = roll(state.current_instruction, 1)
    # display(valhotvec)
    # display(new_top_of_stack)
    topscaled = valhotvec * new_top_of_stack'
    stackscaled = state.stack .* (1.f0 .- new_top_of_stack')
    # new_stack = state.stack .* (1.f0 .- new_top_of_stack') .+ valhotvec * new_top_of_stack'
    new_stack = stackscaled .+ topscaled
    VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
end

###############################
# Utility functions 
###############################

function valhot(val, allvalues)
    [i == val ? 1.0f0 : 0.0f0 for i in allvalues] |> device
end

# Use circshift instead roll?
# Use cumsum (!) instead of sum
function roll(a::Union{CuArray,Array}, increment)
    increment = increment % length(a)
    if increment == 0
        return a
    elseif increment < 0
        return vcat(a[1 - increment:end], a[1:-increment])
    else
        return vcat(a[end + increment - 1:end], a[1:end - increment])
    end 
end

function check_state_asserts(state)
    @assert sum(state.current_instruction) == 1.0
    @assert sum(state.top_of_stack) == 1.0
    for col in eachcol(state.stack)
        @assert sum(col) == 1.0
    end
end

function assert_no_nans(state::VMState)
    @assert !any(isnan.(state.stack)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.top_of_stack)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.current_instruction)) ## Damn, putting an assert removes the NaN
end

function softmaxmask(mask, prog)
    tmp = softmax(prog)
    trainable = tmp .* mask
    frozen = prog .* (1 .- mask)
    trainable .+ frozen
end

function normit(a::Union{Array,CuArray}; dims=1, ϵ=epseltype(a))
    new = a .+ ϵ
    new ./ sum(new, dims=dims) 
end

function normit(a::VMState; dims=1)
    VMState(
        normit(a.current_instruction, dims=dims),
        normit(a.top_of_stack, dims=dims),
        normit(a.stack, dims=dims),
    )
end

function main()
    state = init_state(stackdepth, programlen)
    state = run(state, program, instructions, max_ticks)
    collapsed_program = onecold(program)
end

function applyfullmask(mask, prog)
    out = prog[mask]
    reshape(out, (size(prog)[1], :))
end

###############################
# Initialization functions
###############################

function create_random_discrete_program(len, instructions)
    program = [rand(instructions) for i in 1:len]
end

function create_random_inputs(len, instructions)
    program = [rand(instructions) for i in 1:len]
end

function create_trainable_mask(programlen, inputlen)
    mask = falses(programlen)
    mask[inputlen + 1:end] .= true
    mask
end


function init_state(stackdepth, programlen, allvalues)
    stack = zeros(Float32, length(allvalues), stackdepth)
    current_instruction = zeros(Float32, programlen, )
    top_of_stack = zeros(Float32, stackdepth, )
    stack[1,:] .= 1.f0
    current_instruction[1] = 1.f0
    top_of_stack[1] = 1.f0
    # @assert isbitstype(stack) == true
    state = VMState(
        current_instruction |> device,
        top_of_stack |> device,
        stack |> device,
    )
    state
end

function get_program_with_random_inputs(program, mask)
    num_instructions = size(program)[1]
    new_program = copy(program)
    for (i, col) in enumerate(eachcol(new_program[:, mask]))
        new_program[:,i] .= false
        new_program[rand(1:num_instructions),i] = true
    end
    new_program
end

function create_program_batch(startprogram, trainmask, batch_size)
    target_program = program()
    for i in 1:batch_size
        program
    end
end

function create_examples(hiddenprogram, trainmaskfull; numexamples=16)
    variablemasked = (1 .- trainmaskfull) .* hiddenprogram
    # variablemaskeds = Array{Float32}(undef, (size(variablemasked)..., numexamples))
    variablemaskeds = Array{Float32}(undef, (size(variablemasked)..., numexamples))
    for i in 1:numexamples
        newvariablemasked = copy(variablemasked)
        for col in eachcol(newvariablemasked)
            shuffle!(col)
        end
        variablemaskeds[:,:,i] = newvariablemasked  # Batch
    end
    variablemaskeds
end

function run(state, program, instructions, ticks)
    for i in 1:ticks
        state = super_step(state, program, instructions)
        # assert_no_nans(state)
    end
    state
end

function loss(ŷ, y)
    # TODO top of stack super position actual stack. add tiny amount of current_instruction just because
    # Technically current instruction doesn't really matter
    crossentropy(ŷ.stack, y.stack) +
    crossentropy(ŷ.top_of_stack, y.top_of_stack) +
    crossentropy(ŷ.current_instruction, y.current_instruction)
end

function accuracy(hidden, target, trainmask)
    samemax = onecold(hidden) .== onecold(target)
    result = (sum(samemax) - sum(1 .- trainmask)) / sum(trainmask)
end

function test(hiddenprogram, targetprogram, blank_state, instructions, programlen)
    program = softmaxprog(hiddenprogram)
    target = run(blank_state, targetprogram, instructions, programlen)
    prediction = run(blank_state, program, instructions, programlen)
    loss(prediction, target)
end

function forward(state, target, instructions, programlen, hiddenprogram)
    program = softmaxprog(hiddenprogram)
    pred = run(state, program, instructions, programlen)
    loss(pred, target)
end

function trainloop(variablemaskeds; batchsize=4) 
    # TODO make true function without globals
    # (xbatch, ybatch)
    # grads = applyfullmaskprog(gradprog(data[1][1]))
    # hiddenprograms = varmasked .+ trainablemasked 
    # targetprograms = varmasked .+ targetmasked 
    grads = zeros(Float32, size(applyfullmaskprog(hiddenprogram)))
    @showprogress for i in 1:size(variablemaskeds)[3]
        newgrads = gradprog(hiddenprogram)
        grads = grads .+ applyfullmaskprog(newgrads)
        if i > 0 & i % batchsize == 0 
            Optimise.update!(opt, trainablemasked, grads)
            grads .= 0
        end
    end
end

function trainloopsingle(hiddenprogram; numexamples=4) # TODO make true function without globals
    @showprogress for i in 1:numexamples
        grads = gradprogpart(hiddenprogram)[end]
        grads = applyfullmasktohidden(grads)
        Optimise.update!(opt, hiddenprogram, grads)
    end
end

function trainbatch!(data; batchsize=8) # TODO make true function without globals
    local training_loss
    grads = zeros(Float32, size(hiddenprogram[0]))
    @showprogress for d in data
        # TODO split hiddenprogram from data ?
        newgrads = gradprogpart(hiddenprogram)[end]
        grads = grads .+ applyfullmasktohidden(newgrads)
        if i > 0 & i % batchsize == 0 
            Optimise.update!(opt, hiddenprogram, grads)
            grads .= 0
        end
    end
end

######################################
# Global initialization
######################################
args = Args()

# use_cuda = false
if args.usegpu
    global device = gpu
    # @info "Training on GPU"
else
    global device = cpu
    # @info "Training on CPU"
end

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
blank_state2 = init_state(args.stackdepth, args.programlen, allvalues)


target = run(blank_state, target_program, instructions, args.programlen)

gradprogpart = partial(gradient, forward, blank_state, target, instructions, args.programlen) # Partial?

first_program = deepcopy(program)
# opt = ADAM(0.002) 
opt = Descent(0.1) 

######################################
# Run program train
######################################
first_loss = test(hiddenprogram, target_program, blank_state, instructions, args.programlen)
first_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)

# @time trainloopsingle(hiddenprogram, numexamples=10)

second_loss = test(hiddenprogram, target_program, blank_state, instructions, args.programlen)
second_accuracy = accuracy(hiddenprogram |> cpu, target_program |> cpu, trainmask |> cpu)
@show second_loss - first_loss
@show first_accuracy
@show second_accuracy


