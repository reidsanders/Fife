#using Pkg
#Pkg.activate("fife_env")
using Flux: onehot
using Debugger
println("Running fife")

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

function super_step(state::VMState, instructions)
    new_states = []
    for instruction in instructions
        push!(new_states, instruction(state))
    end
    merge_states(new_states)
end

#function merge_states(states::Vector{VMState})
function merge_states(states)
    new_state = states[1]
    for state in states[2:end]
        new_state.current_instruction .+ state.current_instruction
    end
    new_state
end

instr_pass(state::VMState) = state
instr_5(state::VMState) = instr_val(state,5,allvalues) # TODO create lamdas for all

function roll(a, increment, dims=1)

end

function instr_val(state::VMState, val, allvalues)
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time

    new_stack = state.stack .* onehot(val, allvalues)
    new_stack = softmax(new_stack, dims=1)
    #pushfirst!(state.current_instruction, 0.0)
    #pop!(state.current_instruction)

    pushfirst!(state.top_of_stack, 0.f0)
    #println(new_top_of_stack)
    pop!(state.top_of_stack)
    #println(new_top_of_stack)
    new_top_of_stack = softmax(state.top_of_stack)

    pushfirst!(state.current_instruction, 0.f0)
    pop!(state.current_instruction)
    new_current_instruction = softmax(state.current_instruction)

    new_state = VMState(
        new_current_instruction,
        new_top_of_stack,
        new_stack,
    )
    new_state
end


function main()
    #current_instruction = zeros(Float32, program_len,)::Vector{Float32}
    stack = zeros(length(allvalues), data_stack_depth)
    stack[1,:] .= 1.f0
    state = VMState(
        zeros(Float32,program_len,),
        zeros(Float32,data_stack_depth,),
        stack,
        #softmax(ones(length(allvalues), data_stack_depth), dims=1),

    )
    state.current_instruction[1] = 1.f0
    state.top_of_stack[fld(data_stack_depth,2)] = 1.f0
    #for i in 1:max_ticks
        #state = super_step(state, instructions)
    #end
    state
end

data_stack_depth = 4
program_len = 1
max_ticks = 1
instructions = [instr_pass instr_5]
num_instructions = length(instructions)
# TODO define data possibilities
allvalues = [["blank"]; [i for i in 0:6]]
program = softmax(ones(num_instructions, program_len))


state = @time main()


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


