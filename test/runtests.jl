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
    device,
    StackValue,
    StackFloatType,
    numericvalues,
    nonnumericvalues,
    allvalues,
    ishaltedvalues,
    blanks,
    args

import Fife: instr_pushval!
using Test
using Random
using Flux: onehot, onehotbatch, glorot_uniform, gradient, Descent, Optimise

using ProgressMeter
using CUDA
Random.seed!(123);
CUDA.allowscalar(false)
using Parameters: @with_kw
using Profile
using BenchmarkTools
instr_pushval!(val::StackValue, state::VMState) = instr_pushval!(val, state, allvalues) # TODO remove when types merged?
instr_pushval!(val::Int, state::VMState, allvalues::Array) =
    instr_pushval!(StackValue(val), state, allvalues)
instr_pushval!(val::Int, state::VMState) = instr_pushval!(val, state, allvalues)

function init_random_discretestate(args, allvalues::Array = allvalues)
    state = DiscreteVMState(args)
    state.instrpointer = rand(1:state.programlen)
    rand!(state.input, allvalues)
    rand!(state.output, allvalues)
    rand!(state.stack, allvalues)
    #TODO randomize ishalted or not?
    for key in allvalues
        state.variables[key] = rand(allvalues)
    end
    state
end

function init_random_state(
    stackdepth::Int,
    programlen::Int,
    allvalues::Union{Array,CuArray},
    inputlen::Int,
    outputlen::Int,
)
    instrpointer = zeros(StackFloatType, programlen)
    stackpointer = zeros(StackFloatType, stackdepth)
    inputpointer = zeros(StackFloatType, inputlen)
    outputpointer = zeros(StackFloatType, outputlen)
    ishalted = zeros(StackFloatType, 2)
    input = rand(StackFloatType, length(allvalues), inputlen)
    output = rand(StackFloatType, length(allvalues), outputlen)
    stack = rand(StackFloatType, length(allvalues), stackdepth)
    variables = rand(StackFloatType, length(allvalues), length(allvalues))
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
instructions = [
    [
        instr_pass!,
        instr_halt!,
        instr_pop!,
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
        # instr_load!,
    ]
    val_instructions
]


#############################
# Discrete tests
#############################

function test_instr_halt(args)
    state = DiscreteVMState(args)
    instr_halt!(state)
    @test state.instrpointer == 2
    @test state.ishalted
end

function test_instr_pushval(args)
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    @test state.instrpointer == 2
    @test first(state.stack) == 3
end

function test_instr_pop(args)
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_pop!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 3
end

function test_instr_dup(args)
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_dup!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 5
    popfirst!(state.stack)
    @test first(state.stack) == 5
end

function test_instr_swap(args)
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_swap!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 3
    popfirst!(state.stack)
    @test first(state.stack) == 5
end

function test_instr_add(args)
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_add!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 8
end

function test_instr_sub(args)
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_sub!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 2
end

function test_instr_mult(args)
    state = DiscreteVMState(args)
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
    @test first(state.stack) == args.maxint
end

function test_instr_div(args)
    state = DiscreteVMState(args)
    instr_pushval!(4, state)
    instr_pushval!(7, state)
    instr_div!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 2
end

function test_instr_not(args)
    state = DiscreteVMState(args)
    # test true
    instr_pushval!(3, state)
    instr_not!(state)
    @test state.instrpointer == 3
    @test first(state.stack) == 0
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(0, state)
    instr_not!(state)
    @test state.instrpointer == 3
    @test first(state.stack) == 1
end

function test_instr_and(args)
    state = DiscreteVMState(args)
    # test true
    instr_pushval!(3, state)
    instr_pushval!(1, state)
    instr_and!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(0, state)
    instr_and!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
end

function test_instr_goto(args)
    # test true
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_goto!(state)
    @test state.instrpointer == 6
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(-1, state)
    instr_goto!(state)
    @test state.instrpointer == 3
end

function test_instr_gotoif(args)
    # test true
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_gotoif!(state)
    @test state.instrpointer == 6
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(0, state)
    instr_gotoif!(state)
    @test state.instrpointer == 4
end

function test_instr_iseq(args)
    # test true
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_iseq!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_iseq!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
end

function test_instr_isgt(args)
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_isgt!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
    # test true
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(6, state)
    instr_isgt!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
end

function test_instr_isge(args)
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_isge!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 0
    # test false
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_isge!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
    # test true
    state = DiscreteVMState(args)
    instr_pushval!(3, state)
    instr_pushval!(6, state)
    instr_isge!(state)
    @test state.instrpointer == 4
    @test first(state.stack) == 1
end

function test_instr_store(args)
    # test true
    state = DiscreteVMState(args)
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_store!(state)
    @test state.instrpointer == 4
    @test state.variables[6] == 3
end

function test_instr_load(args)
    # test true
    state = DiscreteVMState(args)
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
    state = DiscreteVMState(args)
    pushfirst!(state.input, 3)
    pushfirst!(state.input, 5)
    instr_read!(state)

    @test state.instrpointer == 2
    @test first(state.stack) == 5
    @test first(state.input) == 3
end

function test_instr_write(args)
    state = DiscreteVMState(args)
    pushfirst!(state.stack, 3)
    pushfirst!(state.stack, 5)
    instr_write!(state)

    @test state.instrpointer == 2
    @test first(state.stack) == 3
    @test first(state.output) == 5
end



function test_convert_discrete_to_continuous(args)
    contstate =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    state = DiscreteVMState(args)
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
    contstate =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    discretestate = DiscreteVMState(args)
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
    # x = [0.0, 0.1, 0.9, 0.0, 0.0]
    # y = [0.0, .7, .3, 0.0, 0.0]
    # result = op_probvec(+, x, y; numericvalues = StackValue.([-args.maxint, 0, 1, 2, args.maxint]))
    # @test result == [0.0, 0.0, 0.1 * 0.7, 0.1 * 0.3 + 0.7 * 0.9, 0.3 * 0.9]
    x = [0.3, 0.7]
    y = [1.0, 0.0]
    result = op_probvec(+, x, y; values = StackValue.([0, 1]))
    @test sum(result) == 1.0
    @test result == [0.3, 0.7]

    x = [0.1, 0.0, 0.1, 0.8, 0.0]
    y = [0.3, 0.0, 0.4, 0.3, 0.0]
    result = op_probvec(
        +,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, 0, 1, args.maxint]),
    )
    target = [1 - ((1 - x[1]) * (1 - y[1])), 0.0, 0.1 * 0.4, 0.1 * 0.3 + 0.4 * 0.8, 0.0]
    @test sum(target) + (0.8 * 0.3) ≈ 1
    @test result == target
end

function test_div_probvec(args)
    x = [0.0, 0.0, 0.1, 0.9, 0.0]
    y = [0.0, 0.0, 0.7, 0.3, 0.0]
    result = op_probvec(
        /,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, 0, 1, args.maxint]),
    )
    @test sum(result) == 1
    @test result == [0.0, 0.0, 0.1 * 0.7 + 0.1 * 0.3, 0.9 * 0.3, 0.9 * 0.7]

    x = [0.0, 0.15, 0.15, 0.7, 0.0]
    y = [0.0, 0.16, 0.0, 0.56, 0.28]
    result =
        op_probvec(/, x, y; values = StackValue.([-args.maxint, -1, 0, 1, args.maxint]))
    @test sum(result) == 1
    @test result[end] == 0
    @test result[1] == 0
    # prob of -args.maxint: sum -args.maxint/0 -args.maxint/1 args.maxint/-1 -1/0
    # @test result[1] == (x[1] * y[2]) + (x[1] * y[3]) + (x[5] * y[3]) + (x[2] * y[3])

    x = [0.0, 0.0, 0.15, 0.15, 0.7, 0.0]
    y = [0.0, 0.0, 0.16, 0.0, 0.56, 0.28]
    result = op_probvec(
        /,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, -1, 0, 1, args.maxint]),
    )
    @test sum(result) == 1
    @test result[end] == 0
    @test result[1] == 0
    @test result[2] == 0
    # prob of -args.maxint: sum -args.maxint/0 -args.maxint/1 args.maxint/-1 -1/0
    # @test result[2] == (x[1] * y[2]) + (x[1] * y[3]) + (x[5] * y[3]) + (x[2] * y[3])

    x = [0.0, 0.2, 0.15, 0.15, 0.4, 0.1]
    y = [0.0, 0.06, 0.16, 0.14, 0.36, 0.28]
    result = op_probvec(
        /,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, -1, 0, 1, args.maxint]),
    )
    @test sum(result) == 1
    @test result[1] == 0
    # prob of -args.maxint: sum -args.maxint/0 -args.maxint/1 args.maxint/-1 -1/0
    @test result[2] == (x[2] * y[4]) + (x[2] * y[5]) + (x[6] * y[3]) + (x[3] * y[4])

    ####
    x = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
    y = [0.0, 0.03, 0.13, 0.23, 0.33, 0.28]
    result = op_probvec(
        /,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, -1, 0, 1, args.maxint]),
    )
    @test sum(result) == 1.0
    # prob of -args.maxint: sum -args.maxint/0 -args.maxint/1 args.maxint/-1 -1/0
    @test result[2] == (0.05 * 0.23) + (0.05 * 0.33) + (0.5 * 0.13) + (0.1 * 0.23)
    ### Prob of -1: sum -1/1 1/-1, -args.maxint / args.maxint, args.maxint/ -args.maxint
    @test result[3] == (0.1 * 0.33) + (0.2 * 0.13) + (0.05 * 0.28) + (0.5 * 0.03)

    ### Test with nonnumeric values
    x = [0.1, 0.05, 0.1, 0.05, 0.2, 0.5]
    y = [0.05, 0.03, 0.13, 0.18, 0.33, 0.28]
    result = op_probvec(
        /,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, -1, 0, 1, args.maxint]),
    )
    @test sum(result) == 1.0
    ### nonumeric prob
    @test result[1] == 0.1 + 0.05 - (0.1 * 0.05)
    # prob of -args.maxint: sum -args.maxint/0 -args.maxint/1 args.maxint/-1 -1/0
    @test result[2] == (0.05 * 0.18) + (0.05 * 0.33) + (0.5 * 0.13) + (0.1 * 0.18)
    ### Prob of -1: sum -1/1 1/-1, -args.maxint / args.maxint, args.maxint/ -args.maxint
    @test result[3] == (0.1 * 0.33) + (0.2 * 0.13) + (0.05 * 0.28) + (0.5 * 0.03)
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = op_probvec(
        /,
        x,
        y;
        values = StackValue.([StackValue(), -args.maxint, -1, 0, 1, args.maxint]),
    )
    @test sum(result) == 1.0
    @test result[1] == 1.0
end

function test_pop_vmstate(args)
    state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newval = valhot(2, allvalues)
    newstate = pushtostack(state, newval)
    newstate, popval = popfromstack(newstate)
    @test newval == popval
    @test newstate.stack == state.stack
end

function test_popfrominput(args)
    state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newinput = fillinput([2, 5, 3], args.inputlen)
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
    state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newval = valhot(2, allvalues)
    state = pushtostack(state, newval)
    @test state.stack[:, end] == newval
end

function test_pushtooutput(args)
    state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    newval = valhot(2, allvalues)
    state = pushtooutput(state, newval)
    @test state.output[:, end] == newval
end

function run_equality_test(x::DiscreteVMState, y::DiscreteVMState)
    @test x.instrpointer == y.instrpointer
    @test x.input == y.input
    @test x.output == y.output
    @test x.stack == y.stack
    for (k, v) in x.variables
        @test v == y.variables[k]
    end
    for (k, v) in y.variables
        @test v == x.variables[k]
    end
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
        test_interpreter_equivalence(args, [instr])
        test_interpreter_equivalence_random_inputs(args, [instr])
    end
end

function test_random_programs(args)
    program = create_random_discrete_program(rand(2:10), instructions)
    test_interpreter_equivalence(args, program)
    test_interpreter_equivalence_random_inputs(args, program)
end

function test_interpreter_equivalence(args, program)
    ### Basic well behaved program ###
    for vals in [
        [],
        [0],
        [3],
        [1, 3, 2, 4],
        [1, 3, 2, 3, 4],
        [1, 3, 2, 4, 0, 1, 3, 3, 4, 2, 1, 2, 3],
        [-3],
        [-1, -3, -2, -3, -4],
        [1, 3, 2, -4, 0, 1, -3, 3, 4, 0, -3],
    ]
        # @info "Test program conversion vals" vals
        contstate = VMState(
            args.stackdepth,
            args.programlen,
            allvalues,
            args.inputlen,
            args.outputlen,
        )
        discretestate = DiscreteVMState(args)
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
        run_equality_test(newcontstate, contstate)
        run_equality_test(newdiscretestate, discretestate)
    end
end

function test_interpreter_equivalence_random_inputs(args, program)
    Random.seed!(123)
    for x = 1:5
        discretestate = init_random_discretestate(args)
        contstate = convert_discrete_to_continuous(discretestate)
        newdiscretestate = convert_continuous_to_discrete(contstate, allvalues)
        run_equality_test(newdiscretestate, discretestate)
        for instr in program
            contstate = instr(contstate)
            instr(discretestate)
        end
        contstate = normalize_stackpointer(contstate)
        contstate = normalize_iopointers(contstate)
        newdiscretestate = convert_continuous_to_discrete(contstate, allvalues)
        newcontstate = convert_discrete_to_continuous(discretestate, allvalues)
        run_equality_test(newcontstate, contstate)
        run_equality_test(newdiscretestate, discretestate)
    end
end


function test_super_step(args)
    ### TODO test super_step / run.
    discrete_program = create_random_discrete_program(args.programlen, instructions)
    program = convert(Array{Float64}, onehotbatch(discrete_program, instructions))
    rand!(program)
    program = normit(program)
    state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
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
            instr_pop!,
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
            # instr_load!,
            instr_read!,
            instr_write!,
        ]
        val_instructions
    ]



    discrete_program = create_random_discrete_program(args.programlen, instructions)
    target_program =
        convert(Array{StackFloatType}, onehotbatch(discrete_program, instructions))
    trainmask = create_trainable_mask(args.programlen, args.inputlen)
    hiddenprogram = deepcopy(target_program)
    hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))

    # Initialize

    trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device

    hiddenprogram = hiddenprogram |> device
    program = softmaxmask(trainmaskfull, hiddenprogram) |> device
    target_program = target_program |> device
    hiddenprogram = hiddenprogram |> device
    trainmask = trainmask |> device

    blank_state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    check_state_asserts(blank_state)
    target = runprogram(blank_state, target_program, instructions, 1000)
end

function test_train(args)
    discrete_program = create_random_discrete_program(args.programlen, instructions)

    target_program =
        convert(Array{StackFloatType}, onehotbatch(discrete_program, instructions))
    trainmask = create_trainable_mask(args.programlen, args.inputlen)
    hiddenprogram = deepcopy(target_program)
    hiddenprogram[:, trainmask] = glorot_uniform(size(hiddenprogram[:, trainmask]))

    # Initialize
    trainmaskfull = repeat(trainmask', outer = (size(hiddenprogram)[1], 1)) |> device
    hiddenprogram = hiddenprogram |> device
    target_program = target_program |> device
    hiddenprogram = hiddenprogram |> device
    trainmask = trainmask |> device

    # Initialize start state with input
    state =
        VMState(args.stackdepth, args.programlen, allvalues, args.inputlen, args.outputlen)
    startstate = VMState(
        state.instrpointer,
        state.stackpointer,
        state.inputpointer,
        state.outputpointer,
        fillinput([2, 5, 3], args.inputlen),
        state.output,
        state.stack,
        state.variables,
        state.ishalted,
    )
    check_state_asserts(startstate)
    target = runprogram(startstate, target_program, instructions, 10)


    ######################################
    # run program train
    ######################################
    first_loss = test(
        hiddenprogram,
        target_program,
        startstate,
        instructions,
        args.programlen,
        trainmaskfull,
    )
    grads = gradient(
        forward,
        startstate,
        target,
        instructions,
        args.programlen,
        hiddenprogram,
        trainmaskfull,
    )
    opt = Descent(0.000001)
    @showprogress for i = 1:3
        grads = gradient(
            forward,
            startstate,
            target,
            instructions,
            args.programlen,
            hiddenprogram,
            trainmaskfull,
        )[end-1]
        grads = grads .* trainmaskfull
        Optimise.update!(opt, hiddenprogram, grads)
    end

    second_loss = test(
        hiddenprogram,
        target_program,
        startstate,
        instructions,
        args.programlen,
        trainmaskfull,
    )
    @info "test train loss improvement" second_loss - first_loss
    @test second_loss < first_loss
end

function test_all_gradient_single_instr(args)
    for instr in instructions
        test_gradient_single_instr(args, instr)
    end
end

function test_gradient_single_instr(args, instr)
    blank_state_random = init_random_state(
        args.stackdepth,
        args.programlen,
        allvalues,
        args.inputlen,
        args.outputlen,
    )
    blank_state_random2 = init_random_state(
        args.stackdepth,
        args.programlen,
        allvalues,
        args.inputlen,
        args.outputlen,
    )
    grad_instr =
        gradient((x, y) -> loss(instr(x), y), blank_state_random, blank_state_random2)
    # try
    #     grad_instr = gradient((x,y) -> loss(instr(x), y), blank_state_random, blank_state_random2)
    # catch e
    #     println("$instr test_gradient_single_instr. Exception: \n $e")
    # end
    @test true
end

function test_gradient_op_probvec(args)

    function optablesingle2(op; numericvalues = numericvalues)
        optable = op.(numericvalues)
        [findall(x -> x == numericval, optable) for numericval in numericvalues]
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

    startstate = init_random_state(
        args.stackdepth,
        args.programlen,
        allvalues,
        args.inputlen,
        args.outputlen,
    )
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

function test_stackvaluetype(args)
    # include("../src/types.jl")
    a = 3
    b = 5
    c = -5
    anew = StackValue(a)
    bnew = StackValue(b)
    cnew = StackValue(c)
    amax = StackValue(args.maxint + 1)
    amin = StackValue(-args.maxint - 1)
    azero = StackValue(0)
    ablank = StackValue()
    @test a == anew
    @test a * b == anew * bnew
    @test a + b == anew + bnew
    @test a - b == anew - bnew
    @test a - bnew == anew - bnew
    @test a + bnew == anew + bnew
    @test a * bnew == anew * bnew
    @test amax + a == amax
    @test amax * a == amax
    @test amax - a == amax
    @test amax * -1 == amin
    @test amax * amin == amin
    @test amax + amin == 0
    @test ablank + amin == ablank
    @test amax * 0 == 0
    @test amin * 0 == 0
    @test amax / 0 == amax
    @test anew / 0 == amax
    @test amin / 0 == amin
    @test cnew / 0 == amin
    @test anew / cnew == round(a / c)
    @test cnew / anew == round(c / a)
    @test cnew / bnew == round(c / b)
    @test amin / anew == amin
    @test amax / anew == amax
    @test amax / cnew == amin
    @test amin / cnew == amax
    @test amax / anew == amax
    @test azero / azero == azero
    @test cnew < 0
    @test amax > a
    @test amin < a
    @test !(ablank < a)
end

@testset "StackValue" begin
    test_stackvaluetype(args)
end
@testset "Discrete Instructions" begin
    test_instr_halt(args)
    test_instr_pushval(args)
    test_instr_pop(args)
    test_instr_dup(args)
    test_instr_swap(args)
    test_instr_add(args)
    test_instr_sub(args)
    test_instr_mult(args)
    test_instr_div(args)
    test_instr_not(args)
    test_instr_and(args)
    test_instr_goto(args)
    test_instr_gotoif(args)
    test_instr_iseq(args)
    test_instr_isgt(args)
    test_instr_isge(args)
    test_instr_store(args)
    test_instr_load(args)
    test_instr_read(args)
    test_instr_write(args)
end
@testset "Fife Utilities" begin
    test_push_vmstate(args)
    test_pushtooutput(args)
    test_popfrominput(args)
    test_pop_vmstate(args)
    test_add_probvec(args)
    test_div_probvec(args)
end
@testset "Convert VMs" begin
    test_convert_discrete_to_continuous(args)
    test_convert_continuous_to_discrete(args)
end
@testset "Instructions" begin
    test_all_single_instr(args)
    test_random_programs(args)
end
@testset "Superposition Interpreter steps" begin
    test_super_step(args)
    test_super_run_program(args)
end
@testset "Train and Gradient" begin
    test_all_gradient_single_instr(args)
    test_gradient_op_probvec(args)
    # args.programlen = 5
    # args.programlen = 5
    # args.maxticks = 10
    # args.lr = .1
    test_train(args)
end