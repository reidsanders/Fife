module SuperInterpreter
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

include("utils.jl")
using .Utils: partial

@with_kw mutable struct Args
    stackdepth::Int = 10
    programlen::Int = 5
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 7
    usegpu::Bool = false
end

include("parameters.jl")

struct VMState
    instructionpointer::Union{Array{Float32},CuArray{Float32}}
    stackpointer::Union{Array{Float32},CuArray{Float32}}
    stack::Union{Array{Float32},CuArray{Float32}}
    # TODO initialize to blank state functions here??
    # Invariants here?
end
struct VMSuperStates
    instructionpointers::Union{Array{Float32},CuArray{Float32}}
    stackpointers::Union{Array{Float32},CuArray{Float32}}
    stacks::Union{Array{Float32},CuArray{Float32}} # num instructions x stack
end
# TODO just have an array of structs? maybe easier to make mutable, instructions can directly modify 


# Zygote.@adjoint VMSuperStates(x,y,z) = VMSuperStates(x,y,z), di -> (di.instructionpointers, di.stackpointers, di.stacks)
# Zygote.@adjoint VMState(x,y,z) = VMState(x,y,z), di -> (di.instructionpointer, di.stackpointer, di.stack)
a::Number * b::VMState = VMState(a * b.instructionpointer, a * b.stackpointer, a * b.stack)
a::VMState * b::Number = b * a
a::VMState + b::VMState = VMState(a.instructionpointer + b.instructionpointer, a.stackpointer + b.stackpointer, a.stack + b.stack)
a::VMState - b::VMState = VMState(a.instructionpointer - b.instructionpointer, a.stackpointer - b.stackpointer, a.stack - b.stack)

length(a::VMSuperStates) = size(a.instructionpointers)[3]
a::Union{Array,CuArray} * b::VMSuperStates = VMSuperStates(a .* b.instructionpointers, a .* b.stackpointers, a .* b.stacks)
a::VMSuperStates * b::Union{Array,CuArray} = b * a

function super_step(state::VMState, program, instructions)
    # TODO instead of taking a state, take the separate arrays as args? Since CuArray doesn't like Structs
    # TODO batch the individual array (eg add superpose dimension -- can that be a struct or needs to be separate?)
    newstates = [instruction(state) for instruction in instructions]
    instructionpointers = cat([x.instructionpointer for x in newstates]..., dims=3)
    stackpointers = cat([x.stackpointer for x in newstates]..., dims=3)

    stacks = cat([x.stack for x in newstates]..., dims=3)

    states = VMSuperStates(
        instructionpointers,
        stackpointers,
        stacks,
    )
    current = program .* state.instructionpointer'
    summed = sum(current, dims=2) 
    summed = reshape(summed, (1, 1, :))
    scaledstates = summed * states 
    reduced = VMState( 
        sum(scaledstates.instructionpointers, dims=3)[:,:,1],
        sum(scaledstates.stackpointers, dims=3)[:,:,1],
        sum(scaledstates.stacks, dims=3)[:,:,1],
    )
    normit(reduced)
end

###############################
# Instructions
###############################

function instr_dup(state::VMState)
    #= 
    DUP Should duplicate top of stack, and push to top of stack 
    =#
    new_stackpointer = roll(state.stackpointer, -1)
    new_stack = state.stack .* (1.f0 .- state.stackpointer') .+ state.stack .* new_stackpointer'
    new_instructionpointer = roll(state.instructionpointer, 1)

    VMState(
        new_instructionpointer,
        new_stackpointer,
        new_stack,
    )
    
end

function instr_swap(state::VMState)
    #= 
    SWAP Should swap top of stack with second to top of stack
    =#
    new_stackpointer = roll(state.stackpointer, -1)
    new_stack = state.stack .* (1.f0 .- state.stackpointer') .+ state.stack .* new_stackpointer'
    new_instructionpointer = roll(state.instructionpointer, 1)

    VMState(
        new_instructionpointer,
        new_stackpointer,
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
    stackscaled = state.stack .* state.stackpointer'
    probofgoto = 1 .- sum(stackscaled .* zerovec, dims=1)
    firstpoptop = roll(state.stackpointer, 1)
    secondstackscaled = state.stack .* firstpoptop' 
    jumpvalprobs = sum(secondstackscaled, dims=2)

    # Assume ints start at 0 should skip 0?
    # Constraint -- length(intvalues) > length of current instruction ?
    jumpvalprobs = jumpvalprobs[length(nonintvalues) + 1:end]
    # TODO length(instructionpointer) compare and truncate
    jumpvalprobs = jumpvalprobs[1:end]
    currentinstructionforward = (1.f0 - sum(jumpvalprobs)) * roll(state.instructionpointer, 1)
    new_instructionpointer = currentinstructionforward .+ jumpvalprobs[1:length(state.instructionpointer)]
    newtop = roll(firstpoptop, 1)

    # jumpvalprobs[:end]
    # TODO set blank / zero to zero? zero goes to first? greater than length(program) goes to end?
    # for each value 0-max in stack, 
    # (??sum diagonal of equivalent jump locations. eg 0 at x is equivalent to 1 at x+1 on stack)
    # TODO if not int value, set to roll 1 forward?
    # TODO add jumpvalprobs to instructionpointer? Then normalize?

    VMState(
        new_instructionpointer,
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
    # display(state.stackpointer)
    new_stackpointer = roll(state.stackpointer, -1)
    new_instructionpointer = roll(state.instructionpointer, 1)
    # display(valhotvec)
    # display(new_stackpointer)
    topscaled = valhotvec * new_stackpointer'
    stackscaled = state.stack .* (1.f0 .- new_stackpointer')
    # new_stack = state.stack .* (1.f0 .- new_stackpointer') .+ valhotvec * new_stackpointer'
    new_stack = stackscaled .+ topscaled
    VMState(
        new_instructionpointer,
        new_stackpointer,
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
    @assert isapprox(sum(state.instructionpointer), 1.0)
    @assert isapprox(sum(state.stackpointer), 1.0)
    for col in eachcol(state.stack)
        @assert isapprox(sum(col), 1.0)
    end
end

function assert_no_nans(state::VMState)
    @assert !any(isnan.(state.stack)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.stackpointer)) ## Damn, putting an assert removes the NaN
    @assert !any(isnan.(state.instructionpointer)) ## Damn, putting an assert removes the NaN
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
        normit(a.instructionpointer, dims=dims),
        normit(a.stackpointer, dims=dims),
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

function VMState(stackdepth::Int=args.stackdepth, programlen::Int=args.programlen, allvalues::Union{Array,CuArray}=allvalues)
    @show length(allvalues)
    @show length(stackdepth)
    stack = zeros(Float32, length(allvalues), stackdepth)
    instructionpointer = zeros(Float32, programlen, )
    stackpointer = zeros(Float32, stackdepth, )
    stack[1,:] .= 1.f0
    instructionpointer[1] = 1.f0
    stackpointer[1] = 1.f0
    # @assert isbitstype(stack) == true
    state = VMState(
        instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
    )
    state
end

#=
function init_state(stackdepth, programlen, allvalues)
    stack = zeros(Float32, length(allvalues), stackdepth)
    instructionpointer = zeros(Float32, programlen, )
    stackpointer = zeros(Float32, stackdepth, )
    stack[1,:] .= 1.f0
    instructionpointer[1] = 1.f0
    stackpointer[1] = 1.f0
    # @assert isbitstype(stack) == true
    state = VMState(
        instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
    )
    state
end
=# 

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
    # TODO top of stack super position actual stack. add tiny amount of instructionpointer just because
    # Technically current instruction doesn't really matter
    crossentropy(ŷ.stack, y.stack) +
    crossentropy(ŷ.stackpointer, y.stackpointer) +
    crossentropy(ŷ.instructionpointer, y.instructionpointer)
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
begin export 
    instr_val, 
    instr_dup, 
    instr_gotoifnotzerofull, 
    instr_pass, 
    instr_swap, 
    valhot,
    VMState, 
    VMSuperStates, 
    trainbatch!, 
    trainloopsingle, 
    trainloop,
    #init_state,
    forward,
    loss,
    test,
    accuracy,
    run,
    create_examples,
    create_program_batch,
    create_random_discrete_program,
    create_random_inputs,
    create_trainable_mask,
    normit,
    softmaxmask,
    roll,
    check_state_asserts,
    assert_no_nans,
    device
end
end