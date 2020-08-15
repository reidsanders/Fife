using Pkg
Pkg.activate(".")
using Flux: onehot, onehotbatch, crossentropy, logitcrossentropy, glorot_uniform
using Flux
using CUDA
using Zygote
using Random
using LoopVectorization

CUDA.allowscalar(false)
#using Debugger
# println("Running fife")

struct VMState
    #=
    All state for the virtual machine
    =#
    # instruction pointer (prob vector)
    current_instruction::Vector{Float32}
    ## top of stack pointer (prob vector)
    top_of_stack::Vector{Float32}
    ## data stack (stacked prob vectors for each possible datum.)
    stack::Array{Float32}
end

function super_step(state::VMState, program, instructions)
    new_states = [instruction(state) for instruction in instructions]

    # new_states = []
    # for instruction in instructions
    #     push!(new_states, instruction(state))
    # end

    # ps = params(program)
    # gs = gradient(ps) do 
    #     new_state = merge_states(new_states, sum(program .* state.current_instruction', dims=2))
    #     return sum(new_state.stack)
    # end
    new_state = merge_states(new_states, sum(program .* state.current_instruction', dims=2))
end

#function merge_states(states::Vector{VMState})
function merge_states(states, weights)
    # TODO merge states should be the weighted average (based on current instruction? and program! state prob is based on prob of that instr in program)
    new_state = states[1]
    new_state.current_instruction = new_state.current_instruction * weights[1]
    new_state.top_of_stack = new_state.top_of_stack * weights[1]
    new_state.stack = new_state.stack * weights[1]
    for (state, weight) in zip(states[2:end], weights[2:end])
        new_state.current_instruction = new_state.current_instruction .+ (state.current_instruction * weight)
        new_state.top_of_stack = new_state.top_of_stack .+ (state.top_of_stack * weight)
        new_state.stack = new_state.stack .+ (state.stack * weight)
    end

    # new_state.current_instruction = new_state.current_instruction / length(states)
    # new_state.top_of_stack = new_state.top_of_stack / length(states)
    # new_state.stack = new_state.stack / length(states)

    # new_state.current_instruction = softmax(new_state.current_instruction)
    # new_state.top_of_stack = softmax(new_state.top_of_stack)
    # new_state.stack = softmax(new_state.stack)
    check_state_asserts(new_state)
    new_state
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
    if increment == 0
    elseif increment < 0
        new = vcat(a[1-increment:end+increment], a[1:-increment])
    else
        new = vcat(a[end+increment-1:end], a[1+increment:end-increment])
    end 
    return new
end

# immutable ImArray <: AbstractArray
#     data::

function instr_val(state::VMState, val, allvalues)
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time
    # sizehint
    # set return type to force allocation

    # new_stack = similar(state.stack)
    valhot = onehot(val, allvalues) 
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

data_stack_depth = 4
program_len = 1
input_len = 1 # frozen
max_ticks = 1
instructions = [instr_2, instr_5]
# instructions = [instr_pass instr_5]
# instructions = [instr_5]
num_instructions = length(instructions)
# TODO define data possibilities
allvalues = [["blank"]; [i for i in 0:5]]
# program = softmax(ones(num_instructions, program_len))


# TODO set trainable part of program to random initialization
# The true program
# randomly generate instruction list, and translate into onehot superposition.
# So "Trainable program" and "Correct Program". Then just run both.
discrete_program = create_random_discrete_program(program_len, instructions)
target_program = onehotbatch(discrete_program, instructions)
target_program = convert(Array{Float32}, target_program)
program = copy(target_program)
train_mask = create_trainable_mask(program_len, input_len)

#Initialize
program[:, train_mask] = glorot_uniform(size(program[:, train_mask]))
blank_state = init_state(data_stack_depth, program_len, allvalues)
# TODO Do we want it to be possible to move instruction pointer to "before" the input?



# Training
# TODO randomly reset input part of program?
# target = run(blank_state, target_program, instructions, program_len)
# pred = run(blank_state, program, instructions, input_len)
# first_loss = loss(pred.stack, target.stack)
# ps = params(@views program[:, train_mask])
ps = params(program)



# newtrainprogram = get_program_with_random_inputs(program, .~train_mask)
# newtargetprogram = copy(target_program)
# newtargetprogram[:, .~train_mask] = newtrainprogram[:, .~train_mask]

# second_loss

Zygote.@adjoint VMState(x,y,z) = VMState(x,y,z), di -> (di.current_instruction, di.top_of_stack, di.stack)
# new = instr_5(blank_state)
# ps = params(program)
# gs1 = gradient(ps) do 
#     new = instr_5(blank_state)
#     return sum(new.stack)
# end
# gs1

# getstack(state) = state.stack
# getinstr(state) = state.current_instruction
# gettop(state) = state.top_of_stack


sumloss(state) = sum(instr_5(state).stack)
gradient(sumloss, blank_state)



# gs2 = gradient(ps) do 
#     target = super_step(blank_state, target_program, instructions)
#     return sum(target.stack)
# end
# gs2

# gs = gradient(ps) do 
#     target = run(blank_state, target_program, instructions, program_len)
#     pred = run(blank_state, program, instructions, input_len)
#     loss(pred.stack, target.stack)
# end
# gs
# create_program_batch(program, train_mask, 16)

# opt = ADAM()


# collapsed_program = @time main()

# collapsed_program




## Word definitions
#def dup(stack):
    #"""
    #DUP Should duplicate top of stack, and push to top of stack

    #move data stack pointer vector up by 1 ?
    #mult each row by corresponding  data stack vector entry (representing prob)

    #prob of word * (Data_Stack array * Top of Stack pointer vector) + (Data_Stack shifted down 1)

    #W * D * SP + D~1

    #Shift stack Pointer?



    #"""
    #stack

#def swap(stack):
    #"""
    #SWAP Swap top two elements in stack

    #prob_of_word * (Data_Stack * Top of Stack )

    #"""

#def push(stack, value):
    #"""
    #push some value to top of stack

    #Any value that has correct type will be pushed to top

    #make matrix with one hot for value. Mult by top of stack, add to data stack, apply softmax
    #"""
    
    #return stack
    #




# TODO impliment basic commands. Use JAX (?) to make superposition of probabilities, run autograd
# basic stack manipulation commands to start?
# Data stack with single uint8 ?
# Any sequence of words is a program. 
# 
# Forth virtual machine:
# Word dictionary
# Current execution pointer
# Data stack ( max_len, uint8)
# return stack (? stack of command pointers ? ) skip for now
# Heap (? skip for now)
# A program takes in a stack (and top of stack pointer) and outputs a stack and pointer (thus is a pure function)
#
# For branching: Use GOTO? Maybe just SKIP?
# DUP, SWAP, ROT, .
# numbers < > = + - * /
# 

# Superposition
# One hot for all possible entries.
# data stack shape: (len, #data_type)
# program shape: (len, #words)
#
# Should the inputs be initialized to random distributions, or lots of actual onehots?
# How to decide termination? The current instruction vector will be moved down by 1 each tick. But the possibility of GOTO before the current instruction means possibly unlimited ticks. Maybe some balance of probability check? (with exponential increasing cooldown between checks?)
#
# Brute force: num_commands ^ prog_len (maybe pruned based on early stoppage, symmetry?)
# superposition: num_ticks * prog_len * num_commands  (basically prog_len^2)
#
# Instead of "popping" have a data stack pointer 
# Need a fairly large stack, and start in the middle ish?
# Use np.roll ? So as to not lose info off the end? Of course you would get weird behavior if it rolled all the way back around...

# Program (array representing probability of symbols)
# Trainable mask
#

