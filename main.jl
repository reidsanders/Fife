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
import Base: +, -, *, length
using StructArrays
using BenchmarkTools
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using Profile
using DataStructures: Deque#, DefaultDict

include("utils.jl")
using .Utils: partial

include("parameters.jl")

include("discreteinterpreter.jl")
import .DiscreteInterpreter:
    DiscreteVMState,
    instr_pass!,
    instr_halt!,
    instr_pushval!,
    instr_pop!,
    instr_dup!,
    instr_swap!,
    instr_add!,
    instr_sub!,
    instr_mult!,
    instr_div!,
    instr_not!,
    instr_and!,
    instr_goto!,
    instr_gotoif!,
    instr_iseq!,
    instr_isgt!,
    instr_isge!,
    instr_store!,
    instr_load!

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

function instr_dup!(state::VMState)
    #= 
    DUP Should duplicate top of stack, and push to top of stack 
    =#
    new_stackpointer = circshift(state.stackpointer, -1)
    oldcomponent = state.stack .* (1.f0 .- new_stackpointer)'
    newcomponent = circshift(state.stack .* state.stackpointer', (0,-1))
    new_stack = oldcomponent .+ newcomponent
    new_instructionpointer = circshift(state.instructionpointer, 1)
    VMState(
        new_instructionpointer,
        new_stackpointer,
        new_stack,
    )
    
end

function instr_add!(state::VMState)
    #= 
    ADD Should pop top two values, and add them, then push that value to top
    =#
    state, x = pop(state)
    state, y = pop(state)


    # TODO need to add intvalues (eg exclude blank?)
    # blank is equivalent to NaN ? Eg include in addition / multiplication table, but replace results with zero?
    # Actually should replace result with "blank". Or just append same vector??
    resultvec = add_probvec(x, y)
    newstate = push(state, resultvec)
    newinstructionpointer = circshift(state.instructionpointer, 1)
    VMState(
        newinstructionpointer,
        newstate.stackpointer,
        newstate.stack,
    )
    
end

function add_probvec(x::Array, y::Array; intvalues=intvalues)
    additiontable = intvalues .+ intvalues'
    additiontable = replacenans.(additiontable, 0.0)
    additiontable = clamp_to_numeric_range.(additiontable; min=intvalues[2], max=intvalues[end-1])
    
    indexmapping = []
    for numericval in intvalues
        append!(indexmapping, [findall(x -> x == numericval, additiontable)])
    end

    xints = x[end + 1 - length(intvalues):end] # Requires intvalues at end of allvalues
    yints = y[end + 1 - length(intvalues):end]
    additionprobs = xints .* yints'

    numericprobs = []
    for indexes in indexmapping
        append!(numericprobs, sum(additionprobs[indexes]))
    end
    # Non numeric values -- just add prob for each individually? You can't really add them so..
    nonnumericprobs = (x[1:end + 1 - length(intvalues)] .+ y[1:end + 1 - length(intvalues)])/2
    [nonnumericprobs; numericprobs]
end

function replacenans(x, replacement)
    if isnan(x)
        return replacement
    else
        return x
    end
end

function clamp_to_numeric_range(x::Number; min=-Inf, max=Inf)
    if x < min
        return -Inf
    elseif x > max
        return Inf
    else
        return x
    end
end

function pop(state::VMState; blankstack=blankstack)
    #= 
    Regular pop. remove prob vector from stack and return
    =#
    newstackpointer = circshift(state.stackpointer, 1)
    scaledstack = state.stack .* newstackpointer'
    scaledblankstack = blankstack .* (1.f0 .- newstackpointer')
    newstack = scaledstack .+ scaledblankstack
    newstate = VMState(
        state.instructionpointer,
        newstackpointer,
        newstack,
    )
    (newstate, sum(newstack, dims=2))
end

function push(state::VMState, valvec::Array) # TODO add shape info?
    #= 
    Regular push. push prob vector to stack based on current stackpointer prob

    note reverses arg ordering of instr in order to match regular push!
    =#
    new_stackpointer = circshift(state.stackpointer, -1)
    topscaled = valvec * new_stackpointer'
    stackscaled = state.stack .* (1.f0 .- new_stackpointer')
    new_stack = stackscaled .+ topscaled
    VMState(
        new_instructionpointer,
        new_stackpointer,
        new_stack,
    )
end

#= 
SWAP Should swap top of stack with second to top of stack
=#
function instr_swap!(state::VMState)
    new_stackpointer = circshift(state.stackpointer, -1)
    new_stack = state.stack .* (1.f0 .- state.stackpointer') .+ state.stack .* new_stackpointer'
    new_instructionpointer = circshift(state.instructionpointer, 1)

    VMState(
        new_instructionpointer,
        new_stackpointer,
        new_stack,
    )
    
end


    
function instr_gotoiffull!(zerovec, nonintvalues, state::VMState)
    #= 
    GOTO takes top two elements of stack. If top is not zero, goto second element (or end, if greater than program len)

    take stack. apply bitmask for 0 / mult by zero onehot (or select col by index, but slow on gpu)
    1.- , then mult as prob vector. 
    Apply top of stack prob? =#

    # TODO don't recalc 
    stackscaled = state.stack .* state.stackpointer'
    probofgoto = 1 .- sum(stackscaled .* zerovec, dims=1)
    firstpoptop = circshift(state.stackpointer, 1)
    secondstackscaled = state.stack .* firstpoptop' 
    jumpvalprobs = sum(secondstackscaled, dims=2)

    # Assume ints start at 0 should skip 0?
    # Constraint -- length(intvalues) > length of current instruction ?
    jumpvalprobs = jumpvalprobs[length(nonintvalues) + 1:end]
    # TODO length(instructionpointer) compare and truncate
    jumpvalprobs = jumpvalprobs[1:end]
    currentinstructionforward = (1.f0 - sum(jumpvalprobs)) * circshift(state.instructionpointer, 1)
    new_instructionpointer = currentinstructionforward .+ jumpvalprobs[1:length(state.instructionpointer)]
    newtop = circshift(firstpoptop, 1)

    # TODO Stack needs to be shifted too

    # jumpvalprobs[:end]
    # TODO set blank / zero to zero? zero goes to first? greater than length(program) goes to end?
    # for each value 0-max in stack, 
    # (??sum diagonal of equivalent jump locations. eg 0 at x is equivalent to 1 at x+1 on stack)
    # TODO if not int value, set to circshift 1 forward?
    # TODO add jumpvalprobs to instructionpointer? Then normalize?

    VMState(
        new_instructionpointer,
        newtop,
        state.stack,
    )
    
end

# TODO normit after most instr (?)
# TODO def normit for  all zero case

function instr_pass!(state::VMState)
    new_instructionpointer = circshift(state.instructionpointer, 1)
    VMState(
        new_instructionpointer,
        state.stackpointer,
        state.stack,
    )
end

function instr_pushval!(val, state::VMState)::VMState
    # This seems really inefficient...
    # Preallocate intermediate arrays? 1 intermediate state for each possible command, so not bad to allocate ahead of time
    # sizehint
    valhotvec = valhot(val, allvalues) # pass allvalues, and partial? 
    new_stackpointer = circshift(state.stackpointer, -1)
    new_instructionpointer = circshift(state.instructionpointer, 1)
    topscaled = valhotvec * new_stackpointer'
    stackscaled = state.stack .* (1.f0 .- new_stackpointer')
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
#########################
#     Conversion        #
#########################
function convert_discrete_to_continuous(discrete::DiscreteVMState, stackdepth=args.stackdepth, programlen=args.programlen, allvalues::Array=allvalues)::VMState
    contstate = VMState(stackdepth, programlen, allvalues)
    cont_instructionpointer = onehot(discrete.instructionpointer, [i for i in 1:programlen]) * 1.f0 
    discretestack = Array{Any,1}(undef, stackdepth)
    fill!(discretestack, "blank")

    for (i,x) in enumerate(discrete.stack)
        discretestack[i] = x
    end
    cont_stack = onehotbatch(discretestack, allvalues) * 1.f0
    state = VMState(
        cont_instructionpointer |> device,
        contstate.stackpointer |> device,
        cont_stack |> device,
    )
    state
    # NOTE if theres no stackpointer the discrete -> super -> discrete aren't consistent (eg symetric)
    # On the other hand super -> discrete is always an lossy process
end

function convert_continuous_to_discrete(contstate::VMState, stackdepth=args.stackdepth, programlen=args.programlen, allvalues=allvalues)::DiscreteVMState
    instructionpointer = onecold(contstate.instructionpointer)
    stackpointer = onecold(contstate.stackpointer)
    stack = [allvalues[i] for i in onecold(contstate.stack)]
    #variables = onecold(contstate.stack)

    stack = circshift(stack, 1 - stackpointer) # Check if this actually makes sense with circshift
    # Dealing with blanks is tricky. It's not clear what is correct semantically
    newstack = Vector{Int}() # Ugly. shouldn't be necessary, but convert doesn't recognize Int64 as Any
    for x in stack
        if x == "blank"
            break
        else
            push!(newstack, x)
        end
    end
    DiscreteVMState(;instructionpointer = instructionpointer, stack = newstack) 
end

function normalize_stackpointer(state::VMState)
    stackpointermax = onecold(state.stackpointer)
    stack = circshift(state.stack, (0, 1 - stackpointermax)) 
    stackpointer = circshift(state.stackpointer, 1 - stackpointermax)
    state = VMState(
        state.instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
    )
end
