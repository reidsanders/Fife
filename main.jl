using Pkg
Pkg.activate(".")
using Flux: onehot, onehotbatch, onecold, crossentropy, logitcrossentropy, glorot_uniform, mse, epseltype
using Flux
using CUDA
using Zygote
using Random
using LoopVectorization
using Base
import Base: +,-,*
using Debugger

CUDA.allowscalar(false)
# TODO use GPU / Torch tensors for better performance
# TODO use threads (if running program on cpu at least)
struct VMState
    # instruction pointer (prob vector)
    current_instruction::Vector{Float32}
    ## top of stack pointer (prob vector)
    top_of_stack::Vector{Float32}
    ## data stack (stacked prob vectors for each possible datum.)
    stack::Array{Float32}
end

function super_step(state::VMState, program, instructions)
    new_states = [instruction(state) for instruction in instructions]
    # instruction_probs = sum(program .* state.current_instruction',dims=2) 
    # scaled_states = instruction_probs .* new_states
    # summed = reduce(+, scaled_states)
    # normed = norm(summed)
    norm(reduce(+, sum(program .* state.current_instruction',dims=2) .* new_states))
end

instr_pass(state::VMState) = state
instr_0(state::VMState) = instr_val(state,0,allvalues) # TODO create lamdas for all
instr_1(state::VMState) = instr_val(state,1,allvalues) # TODO create lamdas for all
instr_2(state::VMState) = instr_val(state,2,allvalues) # TODO create lamdas for all
instr_3(state::VMState) = instr_val(state,3,allvalues) # TODO create lamdas for all
instr_4(state::VMState) = instr_val(state,4,allvalues) # TODO create lamdas for all
instr_5(state::VMState) = instr_val(state,5,allvalues) # TODO create lamdas for all

function roll!(a::Vector, increment)
    # Only vectors right now
    # use hcat / vcat otherwise?
    if increment < 0
        for i in 1:-increment
            push!(a, a[1])
            popfirst!(a)
        end
    else
        for i in 1:increment
            pushfirst!(a, a[end])
            pop!(a)
        end
    end 
end

function roll(a::Vector, increment)
    # Only vectors right now
    # use hcat / vcat otherwise?
    increment = increment%length(a)
    if increment == 0
        return a
    elseif increment < 0
        return vcat(a[1-increment:end], a[1:-increment])
    else
        return vcat(a[end+increment-1:end], a[1:end-increment])
    end 
end

# immutable ImArray <: AbstractArray
#     data::

function instr_val(state::VMState, val, allvalues)
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time
    # sizehint
    # set return type to force allocation

    # new_stack = similar(state.stack)
    # valhot = convert(Vector{Float32},onehot(val, allvalues))
    valhot = [i == val ? 1.0f0 : 0.0f0 for i in allvalues]
    # for (i, col) in enumerate(eachcol(state.stack))
    #     new_stack[:,i] = (0.0-state.top_of_stack[i]) .* col .+ (state.top_of_stack[i] .* valhot)
    # end
    new_stack = state.stack .* (1.0 .- state.top_of_stack') .+ valhot * state.top_of_stack'

    new_top_of_stack = roll(state.top_of_stack,1)
    new_current_instruction = roll(state.current_instruction,1)
    # new_stack = 1.0 .- state.stack

    # print(new_stack)

    new_state = VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
    # check_state_asserts(new_state)
    new_state
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

# Set trainable parts of program.
function create_trainable_mask(program_len, input_len)
    mask = falses(program_len)
    mask[input_len+1:end] .= true
    mask
end

function create_example_batch(batch_size, program_len, input_len)
    # Create input, output
    #
    for i in 1:batch_size
        create_random_program
    end
end

# TODO terminate all program in Null operator? Early stopping if that last instruction is large percentage?
function run(state, program, instructions, ticks)
    # TODO run (takes program superposition and runs it. returns state?)
    for i in 1:ticks
        state = super_step(state, program, instructions)
        # @assert !any(isnan.(program)) ## Damn, putting an assert removes the NaN
        @assert !any(isnan.(state.stack)) ## Damn, putting an assert removes the NaN
        @assert !any(isnan.(state.top_of_stack)) ## Damn, putting an assert removes the NaN
        @assert !any(isnan.(state.current_instruction)) ## Damn, putting an assert removes the NaN
        @assert !any(isnan.(program)) ## Damn, putting an assert removes the NaN
    end
    state
end

function loss(ŷ, y)
    # TODO loss (takes y ŷ  stacks (?) and )
    # use logitcrossentropy ? 
    crossentropy(ŷ, y, ϵ=2*epseltype(ŷ))
end

function init_state(data_stack_depth, program_len, allvalues)
    stack = zeros(length(allvalues), data_stack_depth)
    current_instruction = zeros(Float32,program_len,)
    top_of_stack = zeros(Float32,data_stack_depth,)
    stack[1,:] .= 1.f0
    state = VMState(
        current_instruction,
        top_of_stack,
        stack,
    )
    state.current_instruction[1] = 1.f0
    # state.top_of_stack[fld(data_stack_depth,2)] = 1.f0
    state.top_of_stack[1] = 1.f0
    state
end
function main()
    # current_instruction = zeros(Float32, program_len,)::Vector{Float32}
    # stack = CuArray(zeros(length(allvalues), data_stack_depth))
    # current_instruction = CuArray(zeros(Float32,program_len,))
    # top_of_stack = CuArray(zeros(Float32,data_stack_depth,))

    # stack = zeros(length(allvalues), data_stack_depth)
    # current_instruction = zeros(Float32,program_len,)
    # top_of_stack = zeros(Float32,data_stack_depth,)
    # stack[1,:] .= 1.f0
    # state = VMState(
    #     current_instruction,
    #     top_of_stack,
    #     stack,
    # )
    # state.current_instruction[1] = 1.f0
    # state.top_of_stack[fld(data_stack_depth,2)] = 1.f0

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

# Change target_program untrainable part
function get_program_with_random_inputs(program, mask)
    # Assumes columns are onehot
    num_instructions = size(program)[1]
    new_program = copy(program)
    for (i,col) in enumerate(eachcol(new_program[:, mask]))
        new_program[:,i] .= false
        new_program[rand(1:num_instructions),i] = true
        # col = rand(size(col))
    end
    new_program
end

function create_program_batch(startprogram, train_mask, batch_size)
    program_batch = 
    target_program = program()
    for i in 1:batch_size
        program
    end
end

data_stack_depth = 5
program_len = 4
input_len = 2 # frozen
max_ticks = 4
instructions = [instr_0, instr_1, instr_2, instr_3, instr_4, instr_5]
# instructions = [instr_3, instr_4, instr_5]
num_instructions = length(instructions)
allvalues = [["blank"]; [i for i in 0:5]]


discrete_program = create_random_discrete_program(program_len, instructions)
target_program = onehotbatch(discrete_program, instructions)
target_program = convert(Array{Float32}, target_program)
hiddenprogram = deepcopy(target_program)
train_mask = create_trainable_mask(program_len, input_len)

#Initialize
function softmaxmask(mask, prog)
    new = softmax(prog) .* mask' + prog .* (.~ mask)' 
end
function partial(f,a...)
    ( (b...) -> f(a...,b...) )
end

function norm(a::Array, dims=1)
    new = relu.(a)
    new = new ./ sum(new, dims=dims)
end

function norm(a::VMState, dims=1)
    VMState(
        norm(a.current_instruction),
        norm(a.top_of_stack),
        norm(a.stack),
    )

end
softmaxprog = partial(softmaxmask, train_mask)

hiddenprogram[:, train_mask] = glorot_uniform(size(hiddenprogram[:, train_mask]))
program = softmaxprog(hiddenprogram)
# program[:, train_mask] = softmax(glorot_uniform(size(program[:, train_mask])))
# program[:, train_mask] = softmax(rand(size(program[:, train_mask])))
blank_state = init_state(data_stack_depth, program_len, allvalues)
# TODO Do we want it to be possible to move instruction pointer to "before" the input?


# ps = params(@views program[:, train_mask])



# newtrainprogram = get_program_with_random_inputs(program, .~train_mask)
# newtargetprogram = copy(target_program)
# newtargetprogram[:, .~train_mask] = newtrainprogram[:, .~train_mask]

Zygote.@adjoint VMState(x,y,z) = VMState(x,y,z), di -> (di.current_instruction, di.top_of_stack, di.stack)
a::Number * b::VMState = VMState(a * b.current_instruction, a * b.top_of_stack, a * b.stack)
a::VMState * b::Number = b * a
a::VMState + b::VMState = VMState(a.current_instruction + b.current_instruction, a.top_of_stack + b.top_of_stack, a.stack + b.stack)
a::VMState - b::VMState = VMState(a.current_instruction - b.current_instruction, a.top_of_stack - b.top_of_stack, a.stack - b.stack)

# sumloss(state) = sum(instr_5(state).stack)
# gradient(sumloss, blank_state)

# sumloss(state) = sum(super_step(state,program,instructions).stack)
# gradient(sumloss, blank_state)

target = run(blank_state, target_program, instructions, program_len)
prediction = run(blank_state, program, instructions, program_len)
# first_loss = crossentropy(prediction.stack, target.stack)
first_loss = loss(prediction.stack, target.stack)
# first_loss = mse(prediction.stack, target.stack)

# # trainable = @views program[:, train_mask]
# ps = params(program)
# # ps = params(trainable)

# gs = gradient(ps) do 
#     # target = run(blank_state, target_program, instructions, program_len)
#     pred = run(blank_state, program, instructions, program_len)
#     # logitcrossentropy(pred.stack, target.stack)
#     crossentropy(pred.stack, target.stack)
#     # mse(pred.stack, target.stack)
# end
# gs[program]

using Flux.Optimise: update!


function forward(state, hiddenprogram, target, instructions, program_len)
    program = softmaxprog(hiddenprogram)
    # program = softmaxmask(train_mask, hiddenprogram)
    pred = run(state, program, instructions, program_len)
    # mse(pred.stack, target.stack)
    # crossentropy(pred.stack, target.stack)
    # logitcrossentropy(pred.stack, target.stack)
    loss(pred.stack, target.stack)
end

gradprog(hidden) = gradient(forward,blank_state,hidden,target,instructions,program_len)[2]
gradprog(hiddenprogram)

first_program = deepcopy(program)
# opt = Descent(0.05) # Gradient descent with learning rate 0.1
opt = ADAM(0.001) # Gradient descent with learning rate 0.1
trainable = @views hiddenprogram[:,train_mask]


@show first_loss
function trainloop()
    for i in 1:1000
        # display(i)
        # display(hiddenprogram)
        update!(opt, trainable, gradprog(hiddenprogram)[:, train_mask])
    end
end
# @enter trainloop()
trainloop()


program = softmaxprog(hiddenprogram)
prediction2 = run(blank_state, program, instructions, program_len)
second_loss = loss(prediction2.stack, target.stack)
display(target_program)
display(first_program)
display(program)
@show second_loss
@show second_loss - first_loss

#TODO why is crossentropy increasing loss
# why is gradient sign neg for both instructions in program (for crossentrop)
# why do both losses have a gradient for first instruction (which is exactly accurate so should be 0!)
# TODO mult by top_of_stack before loss (it is relevant afterall)


# ps = params(program)
# gs2 = gradient(ps) do 
#     pred = run(blank_state, program, instructions, program_len)
#     crossentropy(pred.stack, target.stack)
# end
# gs2[program]


# Switch to straight normed to 1 vs softmax? or keep program as a separate unconstrained value, and apply softmax before pushing it into run? Then use logitcrossentropy
# Eg add hidden program with unconstrained values
# sumloss(state) = sum(super_step(state,program,instructions).stack)
# gradient(sumloss, blank_state)