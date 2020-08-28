using Pkg
Pkg.activate(".")
using Flux: onehot, onehotbatch, onecold, crossentropy, logitcrossentropy, glorot_uniform, mse, epseltype
using Flux
using Flux.Optimise: update!
using CUDA
using Zygote
using Random
using LoopVectorization
using Base
using Debugger
using Random
import Base: +,-,*,length
using StructArrays
using BenchmarkTools
Random.seed!(123);

# CUDA.allowscalar(false)
# TODO use GPU / Torch tensors for better performance
# TODO use threads (if running program on cpu at least)
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

Zygote.@adjoint VMSuperStates(x,y,z) = VMSuperStates(x,y,z), di -> (di.current_instructions, di.top_of_stacks, di.stacks)
Zygote.@adjoint VMState(x,y,z) = VMState(x,y,z), di -> (di.current_instruction, di.top_of_stack, di.stack)
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
    summed = reshape(summed,(1,1,:))
    scaledstates = summed * states 
    reduced = VMState( 
        sum(scaledstates.current_instructions, dims=3)[:,:,1],
        sum(scaledstates.top_of_stacks, dims=3)[:,:,1],
        sum(scaledstates.stacks, dims=3)[:,:,1],
    )
    normit(reduced)
    # normit(reduce(+, sum(program .* state.current_instruction',dims=2) .* new_states))
end

function instr_dup(state::VMState)
    #=
    DUP Should duplicate top of stack, and push to top of stack
    =#
    new_top_of_stack = roll(state.top_of_stack,-1)
    new_stack = state.stack .* (1.f0 .- state.top_of_stack') .+ state.stack .* new_top_of_stack'
    new_current_instruction = roll(state.current_instruction,1)

    VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
    
end

function instr_gotoifnotzero(state::VMState)
    #=
    GOTO takes top two elements of stack. If top is not zero, goto second element (or end, if greater than program len)

    take stack. apply bitmask for 0 / mult by zero onehot (or select col by index, but slow on gpu)
    1.- , then mult as prob vector. 
    Apply top of stack prob?
    
    =#

    # new_top_of_stack = roll(state.top_of_stack,2)
    # new_stack = state.stack .* (1.f0 .- state.top_of_stack') .+ state.stack .* new_top_of_stack'
    # new_current_instruction = roll(state.current_instruction,1)

    zerovec = valhot(0, allvalues) # TODO don't recalc 
    stackscaled = state.stack .* state.top_of_stack'
    probofgoto = 1 .- sum(stackscaled .* zerovec, dims=1)
    firstpoptop = roll(state.top_of_stack,1)
    secondstackscaled = state.stack .* firstpoptop' 
    jumpvalprobs = sum(secondstackscaled, dims=2)

    # Assume ints start at 0 should skip 0?
    # Constraint -- length(intvalues) > length of current instruction ?
    jumpvalprobs = jumpvalprobs[length(nonintvalues)+1:end]
    # TODO length(current_instruction) compare and truncate
    jumpvalprobs = jumpvalprobs[1:end]
    currentinstructionforward = (1.f0 - sum(jumpvalprobs)) * roll(state.current_instruction,1)
    new_current_instruction = currentinstructionforward .+ jumpvalprobs[1:length(state.current_instruction)]

    # jumpvalprobs[:end]
    # TODO set blank / zero to zero? zero goes to first? greater than length(program) goes to end?
    # for each value 0-max in stack, 
    # (??sum diagonal of equivalent jump locations. eg 0 at x is equivalent to 1 at x+1 on stack)
    # TODO if not int value, set to roll 1 forward?
    # TODO add jumpvalprobs to current_instruction? Then normalize?

    VMState(
        state.current_instruction,
        firstpoptop,
        state.stack,
    )
    
end

# TODO normit after most instr (?)
# TODO def normit for  all zero case

function valhot(val, allvalues)
    [i == val ? 1.0f0 : 0.0f0 for i in allvalues] |> device
end

instr_pass(state::VMState) = state
instr_0(state::VMState) = instr_val(state, valhot(0,allvalues)) # TODO create lamdas for all
instr_1(state::VMState) = instr_val(state, valhot(1,allvalues)) # TODO create lamdas for all
instr_2(state::VMState) = instr_val(state, valhot(2,allvalues)) # TODO create lamdas for all
instr_3(state::VMState) = instr_val(state, valhot(3,allvalues)) # TODO create lamdas for all
instr_4(state::VMState) = instr_val(state, valhot(4,allvalues)) # TODO create lamdas for all
instr_5(state::VMState) = instr_val(state, valhot(5,allvalues)) # TODO create lamdas for all

# Use circshift instead ?
# Use cumsum (!) instead of sum
function roll(a::Union{CuArray,Array}, increment)
    increment = increment%length(a)
    if increment == 0
        return a
    elseif increment < 0
        return vcat(a[1-increment:end], a[1:-increment])
    else
        return vcat(a[end+increment-1:end], a[1:end-increment])
    end 
end
function instr_val(state::VMState, valhotvec)
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time
    # sizehint
    # set return type to force allocation
    # display(state.top_of_stack)
    new_top_of_stack = roll(state.top_of_stack,-1)
    new_current_instruction = roll(state.current_instruction,1)
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

function check_state_asserts(state)
    @assert sum(state.current_instruction) == 1.0
    @assert sum(state.top_of_stack) == 1.0
    for col in eachcol(state.stack)
        @assert sum(col) == 1.0
    end
end

function create_random_discrete_program(len, instructions)
    program = [rand(instructions) for i in 1:len]
end

function create_trainable_mask(program_len, input_len)
    mask = falses(program_len)
    mask[input_len+1:end] .= true
    mask
end


# TODO terminate all program in Null operator? Early stopping if that last instruction is large percentage?
function run(state, program, instructions, ticks)
    for i in 1:ticks
        # display(i)
        state = super_step(state, program, instructions)
        # assert_no_nans(state)
    end
    state
end

function loss(ŷ, y)
    # TODO top of stack super position actual stack. add tiny amount of current_instruction just because
    crossentropy(ŷ, y)
end

function init_state(data_stack_depth, program_len, allvalues)
    stack = zeros(Float32,length(allvalues), data_stack_depth)
    current_instruction = zeros(Float32,program_len,)
    top_of_stack = zeros(Float32,data_stack_depth,)
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

function softmaxmask(mask, prog)
    # Zygote.ignore() do
    #     display(prog)
    #     display(mask)
    # end
    tmp = softmax(prog)
    # print(tmp)
    # TODO pass in function?
    # apply mask in function?
    trainable = tmp .* mask
    # tmp = tmp .* mask'
    frozen = prog .* (1 .- mask)
    trainable .+ frozen
    # new = softmax(prog) .* mask' + prog .* (1 .- mask)' 
end
function partial(f,a...)
    ( (b...) -> f(a...,b...) )
end

function normit(a::Union{Array,CuArray}; dims=1, ϵ=epseltype(a))
    new = a .+ ϵ
    new ./ sum(new, dims=dims) 
    # TODO if sum to 0, 
    # new = relu.(a)
    # new = new ./ sum(new, dims=dims)
end

function normit(a::VMState; dims=1)
    VMState(
        normit(a.current_instruction, dims=dims),
        normit(a.top_of_stack, dims=dims),
        normit(a.stack, dims=dims),
    )

end
function main()
    state = init_state(data_stack_depth, program_len)
    state = run(state, program, instructions, max_ticks)
    collapsed_program = onecold(program)
end


function custom_train!(loss, ps, data, opt)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss
    ps = Params(ps)
    for d in data
      gs = gradient(ps) do
        training_loss = loss(d...)
        # Code inserted here will be differentiated, unless you need that gradient information
        # it is better to do the work outside this block.
        return training_loss
      end
      # Insert whatever code you want here that needs training_loss, e.g. logging.
      # logging_callback(training_loss)
      # Insert what ever code you want here that needs gradient.
      # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
      update!(opt, ps, gs)
      # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
  end

function get_program_with_random_inputs(program, mask)
    num_instructions = size(program)[1]
    new_program = copy(program)
    for (i,col) in enumerate(eachcol(new_program[:, mask]))
        new_program[:,i] .= false
        new_program[rand(1:num_instructions),i] = true
    end
    new_program
end

function create_program_batch(startprogram, train_mask, batch_size)
    target_program = program()
    for i in 1:batch_size
        program
    end
end

function assert_no_nans(state::VMState)
    @assert !any(isnan.(state.stack)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.top_of_stack)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.current_instruction)) ## Damn, putting an assert removes the NaN
end

function forward(state, hiddenprogram, target, instructions, program_len)
    program = softmaxprog(hiddenprogram)
    pred = run(state, program, instructions, program_len)
    # scale stack by top_of_stack? Need to use all things to work?
    # TODO shouldn't the top_of_stack / current_instruction still count ????? its in hte calculation
    loss(pred.stack, target.stack) + loss(pred.top_of_stack, target.top_of_stack) + loss(pred.current_instruction, target.current_instruction)
end

function trainloop(numexamples) # TODO make true function without globals
    for i in 1:numexamples
        # display(i)
        # display(hiddenprogram)
        # display(hiddenprogram)
        # update!(opt, hiddenprogram, gradprog(hiddenprogram))
        # update!(opt, trainable, gradprog(hiddenprogram)[:, train_mask])

        # grads = gradprog(hiddenprogram)[:, train_mask]
        grads = gradprog(hiddenprogram)
        grads = applyfullmaskprog(grads)
        # grads = reshape(grads,(size(hiddenprogram)[1],:))
        update!(opt, trainable, grads)
    end
end

function applyfullmask(mask,prog)
    out = prog[trainmaskfull]
    reshape(out,(size(prog)[1],:))
end
applyfullmaskprog(program) = applyfullmask(trainmaskfull, program)

use_cuda = false
# use_cuda = true
if use_cuda
    device = gpu
    # @info "Training on GPU"
else
    device = cpu
    # @info "Training on CPU"
end


data_stack_depth = 5
program_len = 2
input_len = 1 # frozen part
max_ticks = 10
instructions = [instr_0, instr_1, instr_2, instr_3, instr_4, instr_5, instr_dup]
# instructions = [instr_0, instr_1, instr_2, instr_3, instr_4, instr_5]
# instructions = [instr_3, instr_4, instr_5]
num_instructions = length(instructions)
maxint = 5
intvalues = [i for i in 0:maxint]
nonintvalues = ["blank"]
allvalues = [nonintvalues; intvalues]



discrete_program = create_random_discrete_program(program_len, instructions)
target_program = onehotbatch(discrete_program, instructions) 
target_program = convert(Array{Float32}, target_program)
train_mask = create_trainable_mask(program_len, input_len)
hiddenprogram = deepcopy(target_program)
hiddenprogram[:, train_mask] = glorot_uniform(size(hiddenprogram[:, train_mask]))

#Initialize

trainmaskfull = repeat(train_mask', outer=(size(hiddenprogram)[1],1))
softmaxprog = partial(softmaxmask, trainmaskfull |> device)

hiddenprogram = hiddenprogram |> device
program = softmaxprog(hiddenprogram) |> device
target_program = target_program |> device
hiddenprogram = hiddenprogram |> device

train_mask = train_mask |> device


# reshape(x[yrep],(length(y),:))


blank_state = init_state(data_stack_depth, program_len, allvalues)
# TODO Do we want it to be possible to move instruction pointer to "before" the input?

# newtrainprogram = get_program_with_random_inputs(program, .!train_mask)
# newtargetprogram = copy(target_program)
# newtargetprogram[:, .!train_mask] = newtrainprogram[:, .!train_mask]

tmp = instr_gotoifnotzero(blank_state)


# target = run(blank_state, target_program, instructions, program_len)
# prediction = run(blank_state, program, instructions, program_len)
# first_loss = loss(prediction.stack, target.stack)
# @show first_loss
# gradprog(hidden) = gradient(forward,blank_state,hidden,target,instructions,program_len)[2]

# first_program = deepcopy(program)
# # opt = Descent(0.05) # Gradient descent with learning rate 0.1
# opt = ADAM(0.001) # Gradient descent with learning rate 0.1
# trainable = @views hiddenprogram[:,train_mask]



# ps = Params(trainable)

# gs = gradient(ps) do 
#     forward(blank_state,hiddenprogram,target,instructions,program_len)
# end
# function trainloopps(numexamples)
#     for i in 1:numexamples
#         # display(i)
#         # display(hiddenprogram)
#         gs = gradient(ps) do 
#             forward(blank_state,hiddenprogram,target,instructions,program_len)
#         end
#         # gs[program]
#         # display(hiddenprogram)
#         # update!(opt, hiddenprogram, gradprog(hiddenprogram))
#         # update!(opt, trainable, gradprog(hiddenprogram)[:, train_mask])
#         update!(opt, ps, gs)
#     end
# end


# # trainloopps(100)
# @btime trainloop(10)



# program = softmaxprog(hiddenprogram)
# prediction2 = run(blank_state, program, instructions, program_len)
# second_loss = loss(prediction2.stack, target.stack)
# # display(target_program)
# # display(first_program)
# # display(program)
# # @show second_loss
# @show second_loss - first_loss

# # predictionbatch = runprog.([program, program, program])
# #TODO why is crossentropy increasing loss
# # why is gradient sign neg for both instructions in program (for crossentrop)
# # why do both losses have a gradient for first instruction (which is exactly accurate so should be 0!)
# # TODO mult by top_of_stack before loss (it is relevant afterall)

# # TODO top_of_stack isnt used in gradient, so it gets iterate nothing?
# #  ignore(), use Params, dropgrad ? calc loss with current_instruction and top_of_stack?

# function compare_programs(hidden, target, trainmaskfull)
#     # prog = softmaxprog(hidden)[trainmaskfull]
#     # # display(onecold(prog))
#     # samemax = onecold(prog) .== onecold(target[trainmaskfull])
#     # display(samemax)
#     # sum(samemax)
#     (sum(onecold(hiddenprogram) .== onecold(target_program))-sum(1 .- train_mask))/sum(train_mask)
# end
# x = compare_programs(hiddenprogram, target_program, trainmaskfull)
# runprog(prog) = run(blank_state, prog, instructions, program_len)
# x
