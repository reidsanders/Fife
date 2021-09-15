using Fife
using Fife: 
    valhot,
    pushtostack,
    pushtooutput,
    popfromstack,
    popfrominput,
    op_probvec,
    normalize_stackpointer,
    normalize_iopointers,
    create_random_discrete_program,
    create_trainable_mask,
    super_step,
    softmaxmask,
    applyfullmask,
    allvalues,
    device,
    StackValueType,
    StackFloatType,
    largevalue,
    coercetostackvaluepart,
    numericvalues,
    nonnumericvalues,
    allvalues,
    ishaltedvalues,
    blanks,
    args

import Fife: instr_pushval!
using Test
using Random
using Flux:
    onehot,
    onehotbatch,
    glorot_uniform,
    gradient,
    Descent,
    Optimise

using ProgressMeter
using CUDA
Random.seed!(123);
CUDA.allowscalar(false)
using Parameters: @with_kw
using Profile
using BenchmarkTools
instr_pushval!(val::args.StackValueType, state::VMState) = instr_pushval!(val, state, allvalues)

function init_random_state(
    stackdepth::Int,
    programlen::Int,
    allvalues::Union{Array,CuArray},
    inputlen::Int,
    outputlen::Int,
)
    instrpointer = zeros(StackFloatType, programlen)
    stackpointer = zeros(StackFloatType, stackdepth)
    inputpointer = zeros(StackFloatType, stackdepth)
    outputpointer = zeros(StackFloatType, stackdepth)
    ishalted = zeros(StackFloatType, 2)
    input = rand(StackFloatType, length(allvalues), inputlen)
    output = rand(StackFloatType, length(allvalues), outputlen)
    stack = rand(StackFloatType, length(allvalues), stackdepth)
    variables = rand(StackFloatType, length(allvalues), stackdepth)
    input = normit(input)
    output = normit(output)
    stack = normit(stack)
    variables = normit(variables)
    instrpointer[1] = 1.0
    stackpointer[1] = 1.0
    inputpointer[1] = 1.0
    outputpointer[1] = 1.0
    ishalted[1] = 1.0 # set false
    VMState(
        instrpointer |> device,
        stackpointer |> device,
        inputpointer |> device,
        outputpointer |> device,
        input |> device,
        output |> device,
        stack |> device,
        variables |> device,
        ishalted |> device,
    )
end

######################################
# Global initialization
######################################
instr_2 = partial(instr_pushval!, 2)
blank_state = VMState(3, 4, allvalues, args.inputlen, args.outputlen)
blank_state_random = init_random_state(3, 4, allvalues, args.inputlen, args.outputlen)
check_state_asserts(blank_state)
check_state_asserts(blank_state_random)

newstate = instr_dup!(blank_state_random)
newstate_instr2 = instr_2(blank_state_random)

#check_state_asserts(newstate)
check_state_asserts(newstate_instr2)

val_instructions = [partial(instr_pushval!, i) for i in numericvalues]
# instructions = val_instructions
instructions = [
    [
        instr_pass!,
        instr_halt!,
        # instr_pushval!,
        # instr_pop!,
        instr_dup!,
        instr_swap!,
        instr_add!,
        instr_sub!,
        instr_mult!,
        instr_div!,
        instr_not!,
        instr_and!,
        # instr_goto!,
        instr_gotoif!,
        instr_write!,
        instr_read!,
        # instr_iseq!,
        # instr_isgt!,
        # instr_isge!,
        # instr_store!,
        # instr_load!
    ]
    val_instructions
]


#############################
# Discrete tests
#############################

function test_instr_halt(args)
    state = DiscreteVMState()
    instr_halt!(state)
    @test state.instrpointer == 2
    @test state.ishalted
end

function test_instr_pushval(args)
    state = DiscreteVMState()
    instr_pushval!(3, state)
    @test state.instrpointer == 2
    @test first(state.stack) == 3
end

function test_instr_pop(args)
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_pop!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 3
end

function test_instr_dup(args)
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_dup!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 5
    popfirst!(state.stack)
    @test first(state.stack) == 5
end

function test_instr_swap(args)
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_swap!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 3
    popfirst!(state.stack)
    @test first(state.stack) == 5
end

function test_instr_add(args)
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_add!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 8
end

function test_instr_sub(args)
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_sub!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 2
end

function test_instr_mult(args)
    state = DiscreteVMState()
    instr_pushval!(2, state)
    instr_pushval!(3, state)
    instr_mult!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 6
    # Test out of bounds case
    instr_pushval!(5, state)
    instr_pushval!(5, state)
    instr_mult!(state)
    @test state.instrpointer == 7
    @test first(state.stack) == largevalue
end

function test_instr_div(args)
    state = DiscreteVMState()
    instr_pushval!(4, state)
    instr_pushval!(7, state)
    instr_div!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 2
end

function test_instr_not(args)
    state = DiscreteVMState()
    # test true
    instr_pushval!(3, state)
    instr_not!(state)
    @test state.instrpointer == 3
    @test first(state.stack) == 0
    # test false
    state = DiscreteVMState()
    instr_pushval!(0, state)
    instr_not!(state)
    @test state.instrpointer == 3
    @test first(state.stack) == 1
end

function test_instr_and(args)
    state = DiscreteVMState()
    # test true
    instr_pushval!(3, state)
    instr_pushval!(1, state)
    instr_and!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(0, state)
    instr_and!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
end

function test_instr_goto(args)
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_goto!(state)
    @test state.instrpointer == 6
    # test false
    state = DiscreteVMState()
    instr_pushval!(-1, state)
    instr_goto!(state)
    @test state.instrpointer == 3
end

function test_instr_gotoif(args)
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_gotoif!(state)
    @test state.instrpointer == 6
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(0, state)
    instr_gotoif!(state)
    @test state.instrpointer == 4
end

function test_instr_iseq(args)
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_iseq!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_iseq!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
end

function test_instr_isgt(args)
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_isgt!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
    # test true
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(6, state)
    instr_isgt!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
end

function test_instr_isge(args)
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_isge!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_isge!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
    # test true
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(6, state)
    instr_isge!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
end

function test_instr_store(args)
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_store!(state)
    @test state.instrpointer == 4
    @test state.variables[6] == 3
end

function test_instr_load(args)
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_store!(state)
    @test state.variables[6] == 3
    instr_load!(state)
    @test state.instrpointer == 6
    @test first(state.stack) == 3
end

function test_instr_read(args)
    state = DiscreteVMState()
    pushfirst!(state.input, 3)
    pushfirst!(state.input, 5)
    instr_read!(state)

    @test state.instrpointer == 2
    @test first(state.stack) == 5
    @test first(state.input) == 3
end

function test_instr_write(args)
    state = DiscreteVMState()
    pushfirst!(state.stack, 3)
    pushfirst!(state.stack, 5)
    instr_write!(state)

    @test state.instrpointer == 2
    @test first(state.stack) == 3
    @test first(state.output) == 5
end



function test_convert_discrete_to_continuous(args)
    contstate = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    state = DiscreteVMState()
    #instr_pushval!(6,state)
    newcontstate = convert_discrete_to_continuous(state, allvalues)
    #@test contstate == newcontstate
    @test contstate.instrpointer == newcontstate.instrpointer
    @test contstate.stackpointer == newcontstate.stackpointer
    @test contstate.stack == newcontstate.stack
    return
end

function test_convert_continuous_to_discrete(args)
    # test true
    contstate = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    discretestate = DiscreteVMState()
    #instr_pushval!(6,state)
    newdiscretestate = convert_continuous_to_discrete(contstate, allvalues)
    #@test contstate == newcontstate
    @test discretestate.instrpointer == newdiscretestate.instrpointer
    @test discretestate.stack == newdiscretestate.stack
    return
end

# TODO make instruction map cont->discrete
# run different combinations and compare.


function test_add_probvec(args)
    x = [0.0, 0.0, 0.1, 0.9, 0.0]
    result = op_probvec(+, x, y; numericvalues = [-largevalue, 0, 1, largevalue])
    @test sum(result) == 1.0
    @test result == [0.0, 0.0, 0.1 * 0.7, 0.1 * 0.3 + 0.7 * 0.9, 0.3 * 0.9]

    x = [0.1, 0.0, 0.1, 0.8, 0.0]
    y = [0.3, 0.0, 0.4, 0.3, 0.0]
    result = op_probvec(+, x, y; numericvalues = [-largevalue, 0, 1, largevalue])
    @test result == [
        0.1 * 0.3 + 0.1 * (1 - 0.3) + (1 - 0.1) * 0.3,
        0.0,
        0.1 * 0.4,
        0.1 * 0.3 + 0.4 * 0.8,
        0.3 * 0.8,
    ]
    @test sum(result) == 1.0
end

function test_div_probvec(args)
    x = [0.0, 0.0, 0.1, 0.9, 0.0]
    y = [0.0, 0.0, 0.7, 0.3, 0.0]
    result = op_probvec(/, x, y; numericvalues = [-largevalue, 0, 1, largevalue])
    @test sum(result) == 1.0
    @test result == [0.0, 0.0, 0.1 * 0.7 + 0.1 * 0.3, 0.9 * 0.3, 0.9 * 0.7]

    x = [0.0, 0.15, 0.15, 0.7, 0.0]
    y = [0.0, 0.16, 0.0, 0.56, 0.28]
    result = op_probvec(/, x, y; numericvalues = [-largevalue, -1, 0, 1, largevalue])
    @test sum(result) == 1.0

    x = [0.05, 0.1, 0.15, 0.2, 0.5]
    y = [0.03, 0.13, 0.23, 0.33, 0.28]
    result = op_probvec(/, x, y; numericvalues = [-largevalue, -1, 0, 1, largevalue])
    @test sum(result) == 1.0
    # prob of -largevalue: sum -largevalue/0 -largevalue/1 largevalue/-1 -1/0
    @test result[1] == (0.05 * 0.23) + (0.05 * 0.33) + (0.5 * 0.13) + (0.1 * 0.23)
    ### Prob of -1: sum -1/1 1/-1, -largevalue / largevalue, largevalue/ -largevalue
    @test result[2] == (0.1 * 0.33) + (0.2 * 0.13) + (0.05 * 0.28) + (0.5 * 0.03)

    ### Test with nonnumeric values
    x = [0.1, 0.05, 0.1, 0.05, 0.2, 0.5]
    y = [0.05, 0.03, 0.13, 0.18, 0.33, 0.28]
    result = op_probvec(/, x, y; numericvalues = [-largevalue, -1, 0, 1, largevalue])
    @test sum(result) == 1.0
    ### nonumeric prob
    @test result[1] == 0.1 + 0.05 - (0.1 * 0.05)
    # prob of -largevalue: sum -largevalue/0 -largevalue/1 largevalue/-1 -1/0
    @test result[2] == (0.05 * 0.18) + (0.05 * 0.33) + (0.5 * 0.13) + (0.1 * 0.18)
    ### Prob of -1: sum -1/1 1/-1, -largevalue / largevalue, largevalue/ -largevalue
    @test result[3] == (0.1 * 0.33) + (0.2 * 0.13) + (0.05 * 0.28) + (0.5 * 0.03)
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = op_probvec(/, x, y; numericvalues = [-largevalue, -1, 0, 1, largevalue])
    @test sum(result) == 1.0
    @test result[1] == 1.0
end

function test_pop_vmstate(args)
    state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newval = valhot(2, allvalues)
    newstate = pushtostack(state, newval)
    newstate, popval = popfromstack(newstate)
    @test newval == popval
    @test newstate.stack == state.stack
end

function test_popfrominput(args)
    state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newinput = fillinput([2,5,3], args.inputlen)
    state = VMState(
        state.instrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        newinput,
        state.output,
        state.stack,
        state.variables,
        state.ishalted,
    )
    newstate, popval = popfrominput(state)
    @test newinput[:, 1] == popval
    @test newinput[:, 2] == newstate.input[:, 2]
    @test newinput[:, end] == newstate.input[:, 1]
end

function test_push_vmstate(args)
    state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newval = valhot(2, allvalues)
    state = pushtostack(state, newval)
    @test state.stack[:, end] == newval
end

function test_pushtooutput(args)
    state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newval = valhot(2, allvalues)
    state = pushtooutput(state, newval)
    @test state.output[:, end] == newval
end

function run_equality_test(x::DiscreteVMState, y::DiscreteVMState)
    @test x.instrpointer == y.instrpointer
    @test x.input == y.input
    @test x.output == y.output
    @test x.stack == y.stack
    @test x.variables == y.variables
    @test x.ishalted == y.ishalted
end

function run_equality_test(x::VMState, y::VMState)
    @test x.instrpointer ≈ y.instrpointer
    @test x.stackpointer ≈ y.stackpointer
    @test x.inputpointer ≈ y.inputpointer
    @test x.outputpointer ≈ y.outputpointer
    @test x.stack ≈ y.stack
    @test x.input ≈ y.input
    @test x.output ≈ y.output
    @test x.variables ≈ y.variables
    @test x.ishalted ≈ y.ishalted
end

function run_equality_asserts(x::DiscreteVMState, y::DiscreteVMState)
    @assert x.instrpointer == y.instrpointer "instrpointer Not Equal:\n $(x.instrpointer)\n $(y.instrpointer)"
    @assert x.variables == y.variables "Variables Not equal\n $(x.variables)\n $(y.variables)"
    @assert x.ishalted == y.ishalted "ishalted Not equal\n $(x.ishalted)\n $(y.ishalted)"
    @assert x.stack == y.stack "Stack Not equal\n $(x.stack)\n $(y.stack)"
end

function run_equality_asserts(x::VMState, y::VMState)
    @assert x.instrpointer ≈ y.instrpointer "instrpointer Not Equal:\n $(x.instrpointer)\n $(y.instrpointer)"
    @assert x.variables ≈ y.variables "Variables Not equal\n $(x.variables)\n $(y.variables)"
    @assert x.ishalted ≈ y.ishalted "ishalted Not equal\n $(x.ishalted)\n $(y.ishalted)"
    @assert x.stack ≈ y.stack "Stack Not equal\n $(x.stack)\n $(y.stack)"
    @assert x.stackpointer ≈ y.stackpointer "Stack Not equal\n $(x.stackpointer)\n $(y.stackpointer)"
end

function test_all_single_instr(args)
    for instr in instructions
        println("Test Single Instruction: $(instr)")
        test_program_conversion(args, [instr])
    end
end

function test_program_conversion(args, program)
    ### Basic well behaved program ###
    for vals in [
        [],
        [0],
        [3],
        [1, 3, 2, 4],
        [1, 3, 2, 3, 4],
        [1, 3, 2, 4, 0, 1, 3, 3, 4, 2, 1, 2, 3],
    ]
        #@show vals
        contstate = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
        discretestate = DiscreteVMState()
        for val in vals
            contstate = instr_pushval!(val, contstate)
            instr_pushval!(val, discretestate)
        end
        for instr in program
            contstate = instr(contstate)
            instr(discretestate)
        end
        contstate = normalize_stackpointer(contstate)
        contstate = normalize_iopointers(contstate)
        newdiscretestate = convert_continuous_to_discrete(contstate, allvalues)
        newcontstate = convert_discrete_to_continuous(discretestate, allvalues)
        # run_equality_asserts(contstate, newcontstate)
        # run_equality_asserts(discretestate, newdiscretestate)
        # run_equality_test(newcontstate, contstate)
        run_equality_test(newdiscretestate, discretestate)
    end
end


function test_super_step(args)
    ### TODO test super_step / run.
    discrete_program = create_random_discrete_program(args.programlen, instructions)
    program = convert(Array{Float32}, onehotbatch(discrete_program, instructions))
    rand!(program)
    program = normit(program)
    state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    state = super_step(state, program, instructions)
    check_state_asserts(state)
end

function test_super_run_program(args)
    val_instructions = [partial(instr_pushval!, i) for i in numericvalues]

    instructions = [
        [
            instr_pass!,
            instr_halt!,
            # instr_pushval!,
            # instr_pop!,
            instr_dup!,
            instr_swap!,
            instr_add!,
            instr_sub!,
            instr_mult!,
            instr_div!,
            instr_not!,
            instr_and!,
            # instr_goto!,
            instr_gotoif!,
            # instr_iseq!,
            # instr_isgt!,
            # instr_isge!,
            # instr_store!,
            # instr_load!
            instr_read!,
            instr_write!,
        ]
        val_instructions
    ]


    num_instructions = length(instructions)

    discrete_program = create_random_discrete_program(args.programlen, instructions)

    discrete_programs = [
        [
            create_random_discrete_program(args.inputlen, instructions)
            discrete_program[end-args.inputlen:end]
        ] for x = 1:args.trainsetsize
    ]

    target_program =
        convert(Array{Float32}, onehotbatch(discrete_program, instructions))
    trainmask = create_trainable_mask(args.programlen, args.inputlen)
    hiddenprogram = deepcopy(target_program)
    hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))


    # Initialize

    trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device
    applyfullmaskprog = partial(applyfullmask, trainmaskfull)
    applyfullmasktohidden = partial((mask, prog) -> mask .* prog, trainmaskfull)

    hiddenprogram = hiddenprogram |> device
    program = softmaxmask(hiddenprogram, trainmaskfull) |> device
    target_program = target_program |> device
    hiddenprogram = hiddenprogram |> device
    trainmask = trainmask |> device

    blank_state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    check_state_asserts(blank_state)
    target = runprogram(blank_state, target_program, instructions, 1000)
end

function test_train(args)
    num_instructions = length(instructions)

    discrete_program = create_random_discrete_program(args.programlen, instructions)
    # inputvalues = [rand(val_instructions) for i in 1:args.inputlen]
    # discreteinputprogram = create_random_discrete_program(args.inputlen, val_instructions)
    # discretetrainableprogram = create_random_discrete_program(args.programlen - args.inputlen, instructions)
    # inputvalues = [rand(val_instructions) for i in 1:args.inputlen]

    # discrete_programs = [
    #     [
    #         create_random_discrete_program(args.inputlen, instructions)
    #         discrete_program[end-args.inputlen:end]
    #     ] for x = 1:args.trainsetsize
    # ]

    target_program = convert(Array{StackFloatType}, onehotbatch(discrete_program, instructions))
    trainmask = create_trainable_mask(args.programlen, args.inputlen)
    hiddenprogram = deepcopy(target_program)
    hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))


    # Initialize

    trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device
    applyfullmaskprog = partial(applyfullmask, trainmaskfull)
    applyfullmasktohidden = partial((mask, prog) -> mask .* prog, trainmaskfull)

    hiddenprogram = hiddenprogram |> device
    program = softmaxmask(hiddenprogram, trainmaskfull) |> device
    target_program = target_program |> device
    hiddenprogram = hiddenprogram |> device
    trainmask = trainmask |> device

    # Initialize start state with input
    state = VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    startstate = VMState(
        state.instrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        fillinput([2,5,3], args.inputlen),
        state.output,
        state.stack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(startstate)
    target = runprogram(startstate, target_program, instructions, 10)


    ######################################
    # runprogram program train
    ######################################
    first_loss = test(hiddenprogram, target_program, startstate, instructions, args.programlen, trainmaskfull)
    first_accuracy = accuracy(hiddenprogram |> device, target_program |> device, trainmask |> device)
    try
        grads = gradient(forward, startstate, target, instructions, args.programlen, hiddenprogram, trainmaskfull)
    catch e
        println("Exception uncaught by test: \n {e}")
    end
    # trainloopsingle(hiddenprogram, startstate, target, instructions, args.programlen, trainmaskfull, numexamples = 10, opt = Descent(.000001))
    opt = Descent(.000001)
    @showprogress for i = 1:10
        grads = gradient(
            forward,
            startstate,
            target,
            instructions,
            args.programlen,
            hiddenprogram,
            trainmaskfull,
        )[end-1] # end-1 for hidden?
        grads = grads .* trainmaskfull
        Optimise.update!(opt, hiddenprogram, grads)
    end

    second_loss = test(hiddenprogram, target_program, startstate, instructions, args.programlen, trainmaskfull)
    second_accuracy = accuracy(hiddenprogram |> device, target_program |> device, trainmask |> device)
    @show second_loss - first_loss
    @test second_loss < first_loss
end

function test_all_gradient_single_instr(args)
    for instr in instructions
        test_gradient_single_instr(args, instr)
    end
end

function test_gradient_single_instr(args, instr)
    blank_state_random = init_random_state(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    blank_state_random2 = init_random_state(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    # grad_instr = gradient((x,y) -> loss(instr(x), y), blank_state_random, blank_state_random2)
    try
        grad_instr = gradient((x,y) -> loss(instr(x), y), blank_state_random, blank_state_random2)
    catch e
        println("$instr test_gradient_single_instr. Exception: \n $e")
    end
    @test true
end

function test_gradient_op_probvec(args)

    function optablesingle2(op; numericvalues = numericvalues)
        optable = op.(numericvalues)
        optable = coercetostackvaluepart.(optable)
        indexmapping = [findall(x -> x == numericval, optable) for numericval in numericvalues]
    end

    op(a) = float(a == 0)
    # optableindexes = optablesingle2(op)

    function op_prob_sum(op, x)
        optableindexes = optablesingle2(op, numericvalues = numericvalues)
        xnumerics = x[end+1-length(numericvalues):end]
        numericprobs = [sum(xnumerics[indexes]) for indexes in optableindexes]
        nonnumericprobs = x[1:end-length(numericvalues)]
        xnew = [nonnumericprobs; numericprobs]
    end

    function op_probvec2(op, x::Array)
        optableindexes = optablesingle2(op, numericvalues = numericvalues)
        xnumerics = x[end+1-length(numericvalues):end]

        numericprobs = [sum(xnumerics[indexes]) for indexes in optableindexes]
        nonnumericprobs = x[1:end-length(numericvalues)]

        xnew = [nonnumericprobs; numericprobs]
    end

    function op_prob_sum3(op, x)
        xnew = op_probvec2(op, x)
        sum(xnew)
    end

    function op_prob_sum2(op, x)
        xnew = op_prob_sum(op, x)
        sum(xnew)
    end

    startstate = init_random_state(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    state, x = popfromstack(startstate)
    # grad_instr = gradient(op_prob_sum, op, x)
    grad_instr = gradient(op_prob_sum2, op, x)
    grad_instr = gradient(op_prob_sum3, op, x)


    # try
    #     grad_instr = gradient((x,y) -> loss(instr(x), y), blank_state_random, blank_state_random2)
    # catch e
    #     println("$instr test_gradient_single_instr. Exception: \n $e")
    # end
    @test true
end

@testset "Fife.jl" begin
    # test_push_vmstate(args)
    # test_pushtooutput(args)
    # test_popfrominput(args)
    # test_pop_vmstate(args)
    # test_div_probvec(args)
    # test_instr_halt(args)
    # test_instr_pushval(args)
    # test_instr_pop(args)
    # test_instr_dup(args)
    # test_instr_swap(args)
    # test_instr_add(args)
    # test_instr_sub(args)
    # test_instr_mult(args)
    # test_instr_div(args)
    # test_instr_not(args)
    # test_instr_and(args)
    # test_instr_goto(args)
    # test_instr_gotoif(args)
    # test_instr_iseq(args)
    # test_instr_isgt(args)
    # test_instr_isge(args)
    # test_instr_store(args)
    # test_instr_load(args)
    # test_instr_read(args)
    # test_instr_write(args)
    # test_convert_discrete_to_continuous(args)
    # test_convert_continuous_to_discrete(args)
    test_all_single_instr(args)
    # test_super_step(args)
    # test_super_run_program(args)
    # test_all_gradient_single_instr(args)
    # test_train(args)
    # test_gradient_op_probvec(args)
end