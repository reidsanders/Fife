using Pkg
Pkg.activate(".")
using Flux: onehot
using Flux
#using Debugger
println("Running fife")

mutable struct VMState
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

function super_step(state::VMState, instructions)
    new_states = []
    for instruction in instructions
        push!(new_states, instruction(state))
    end
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

function roll(a::Vector, increment)
    # Only vectors right now
    # use hcat / vcat otherwise?
    if increment <= 0
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

function instr_val(state::VMState, val, allvalues)
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time
    # set return type
    new_stack = similar(state.stack)
    valhot = onehot(val, allvalues) 
    for (i, col) in enumerate(eachcol(state.stack))
        new_stack[:,i] = (1.0-state.top_of_stack[i]) .* col .+ (state.top_of_stack[i] .* valhot)
    end
    #new_stack = softmax(new_stack, dims=1)

    new_top_of_stack = copy(state.top_of_stack)
    roll(new_top_of_stack,1)
    new_current_instruction = copy(state.current_instruction)
    roll(new_current_instruction,1)

    new_state = VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
    check_state_asserts(new_state)
    new_state
end

function check_state_asserts(state)

    @assert sum(state.current_instruction) == 1.0
    @assert sum(state.top_of_stack) == 1.0
    for col in eachcol(state.stack)
        @assert sum(col) == 1.0
    end
end


function main()
    #current_instruction = zeros(Float32, program_len,)::Vector{Float32}
    stack = zeros(length(allvalues), data_stack_depth)
    #stack = softmax(stack)
    # stack = zeros(length(allvalues), data_stack_depth)
    stack[1,:] .= 1.f0
    state = VMState(
        zeros(Float32,program_len,),
        zeros(Float32,data_stack_depth,),
        stack,
        #softmax(ones(length(allvalues), data_stack_depth), dims=1),

    )
    state.current_instruction[1] = 1.f0
    state.top_of_stack[fld(data_stack_depth,2)] = 1.f0
    for i in 1:max_ticks
        state = super_step(state, instructions)
    end
    state
end

data_stack_depth = 6
program_len = 7
max_ticks = 2
instructions = [instr_2 instr_5]
# instructions = [instr_pass instr_5]
# instructions = [instr_5]
num_instructions = length(instructions)
# TODO define data possibilities
allvalues = [["blank"]; [i for i in 0:6]]
program = softmax(ones(num_instructions, program_len))


state = @time main()

state.stack




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

