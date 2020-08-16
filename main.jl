using Pkg
Pkg.activate(".")
using Flux: onehot, onehotbatch, crossentropy, logitcrossentropy, glorot_uniform
using Flux
using CUDA
using Zygote
using Random
using LoopVectorization
import Base: +,-,*

CUDA.allowscalar(false)

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
    reduce(+, sum(program .* state.current_instruction') .* new_states)
end

instr_pass(state::VMState) = state
instr_5(state::VMState) = instr_val(state,5,allvalues) # TODO create lamdas for all
instr_2(state::VMState) = instr_val(state,2,allvalues) # TODO create lamdas for all

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
    end
    state
end

function loss(ŷ, y)
    # TODO loss (takes y ŷ  stacks (?) and )
    # use logitcrossentropy ? 
    crossentropy(ŷ, y)
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
    state.top_of_stack[fld(data_stack_depth,2)] = 1.f0
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

data_stack_depth = 10
program_len = 4
input_len = 2 # frozen
max_ticks = 2
instructions = [instr_2, instr_5]
num_instructions = length(instructions)
allvalues = [["blank"]; [i for i in 0:5]]


discrete_program = create_random_discrete_program(program_len, instructions)
target_program = onehotbatch(discrete_program, instructions)
target_program = convert(Array{Float32}, target_program)
program = copy(target_program)
train_mask = create_trainable_mask(program_len, input_len)

#Initialize
program[:, train_mask] = glorot_uniform(size(program[:, train_mask]))
blank_state = init_state(data_stack_depth, program_len, allvalues)
# TODO Do we want it to be possible to move instruction pointer to "before" the input?



ps = params(@views program[:, train_mask])



# newtrainprogram = get_program_with_random_inputs(program, .~train_mask)
# newtargetprogram = copy(target_program)
# newtargetprogram[:, .~train_mask] = newtrainprogram[:, .~train_mask]

Zygote.@adjoint VMState(x,y,z) = VMState(x,y,z), di -> (di.current_instruction, di.top_of_stack, di.stack)
a::Number * b::VMState = VMState(a * b.current_instruction, a * b.top_of_stack, a * b.stack)
a::VMState * b::Number = b * a
a::VMState + b::VMState = VMState(a.current_instruction + b.current_instruction, a.top_of_stack + b.top_of_stack, a.stack + b.stack)
a::VMState - b::VMState = VMState(a.current_instruction - b.current_instruction, a.top_of_stack - b.top_of_stack, a.stack - b.stack)

sumloss(state) = sum(instr_5(state).stack)
gradient(sumloss, blank_state)

sumloss(state) = sum(super_step(state,program,instructions).stack)
gradient(sumloss, blank_state)

target = run(blank_state, target_program, instructions, program_len)
pred = run(blank_state, program, instructions, input_len)

gs = gradient(ps) do 
    target = run(blank_state, target_program, instructions, program_len)
    pred = run(blank_state, program, instructions, input_len)
    sum(pred.stack + target.stack)
    # loss(pred.stack, target.stack)
    logitcrossentropy(pred.stack, target.stack)
end
gs
