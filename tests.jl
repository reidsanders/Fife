using Test
using Flux
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using DataStructures: CircularDeque, DefaultDict
using Flux:
    onehot,
    onehotbatch,
    onecold,
    crossentropy,
    logitcrossentropy,
    glorot_uniform,
    mse,
    epseltype
using CUDA
using Random
import Base: ==

#include("main.jl")
include("fife.jl")

######################################
# Global initialization
######################################
@with_kw mutable struct TestArgs
    batchsize::Int = 2
    lr::Float32 = 2e-4
    epochs::Int = 2
    stackdepth::Int = 5
    programlen::Int = 7
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 8
    usegpu::Bool = false
end
args = TestArgs()
include("parameters.jl")

function init_random_state(stackdepth, programlen, allvalues)
    stack = rand(Float32, length(allvalues), stackdepth)
    instructionpointer = zeros(Float32, programlen)
    stackpointer = zeros(Float32, stackdepth)
    #instructionpointer = rand(Float32, programlen, )
    #stackpointer = rand(Float32, stackdepth, )
    stack = normit(stack)
    instructionpointer[1] = 1.0f0
    stackpointer[1] = 1.0f0
    state = VMState(instructionpointer |> device, stackpointer |> device, stack |> device)
    #normit(state)
end

#allvalues = [i for i in 0:5]
#instr_2 = partial(instr_pushval!, valhot(2, allvalues))
instr_2 = partial(instr_pushval!, 2)

blank_state = VMState(3, 4, allvalues)
blank_state_random = init_random_state(3, 4, allvalues)
check_state_asserts(blank_state)
check_state_asserts(blank_state_random)

newstate = instr_dup!(blank_state_random)
newstate_instr2 = instr_2(blank_state_random)

#check_state_asserts(newstate)
check_state_asserts(newstate_instr2)

#############################
# Discrete tests
#############################

function test_instr_halt()
    state = DiscreteVMState()
    instr_halt!(state)
    @test state.instructionpointer == 2
    @test state.ishalted
end

function test_instr_pushval()
    state = DiscreteVMState()
    instr_pushval!(3, state)
    @test state.instructionpointer == 2
    @test first(state.stack) == 3
end

function test_instr_pop()
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_pop!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 3
end

function test_instr_dup()
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_dup!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 5
    popfirst!(state.stack)
    @test first(state.stack) == 5
end

function test_instr_swap()
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_swap!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 3
    popfirst!(state.stack)
    @test first(state.stack) == 5
end

function test_instr_add()
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_add!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 8
end

function test_instr_sub()
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(5, state)
    instr_sub!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 2
end

function test_instr_mult()
    state = DiscreteVMState()
    instr_pushval!(2, state)
    instr_pushval!(3, state)
    instr_mult!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 6
    # Test out of bounds case
    instr_pushval!(5, state)
    instr_pushval!(5, state)
    instr_mult!(state)
    @test state.instructionpointer == 7
    @test first(state.stack) == Inf
end

function test_instr_div()
    state = DiscreteVMState()
    instr_pushval!(4, state)
    instr_pushval!(9, state)
    instr_div!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 2
end

function test_instr_not()
    state = DiscreteVMState()
    # test true
    instr_pushval!(3, state)
    instr_not!(state)
    @test state.instructionpointer == 3
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(0, state)
    instr_not!(state)
    @test state.instructionpointer == 3
    @test first(state.stack) == 0
end

function test_instr_and()
    state = DiscreteVMState()
    # test true
    instr_pushval!(3, state)
    instr_pushval!(1, state)
    instr_and!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(0, state)
    instr_and!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 0
end

function test_instr_goto()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_goto!(state)
    @test state.instructionpointer == 6
    # test false
    state = DiscreteVMState()
    instr_pushval!(-1, state)
    instr_goto!(state)
    @test state.instructionpointer == 3
end

function test_instr_gotoif()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_gotoif!(state)
    @test state.instructionpointer == 6
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(0, state)
    instr_gotoif!(state)
    @test state.instructionpointer == 4
end

function test_instr_iseq()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_iseq!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_iseq!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 0
end

function test_instr_isgt()
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_isgt!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 0
    # test true
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(6, state)
    instr_isgt!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 1
end

function test_instr_isge()
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_isge!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 0
    # test false
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_isge!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 1
    # test true
    state = DiscreteVMState()
    instr_pushval!(3, state)
    instr_pushval!(6, state)
    instr_isge!(state)
    @test state.instructionpointer == 4
    @test first(state.stack) == 1
end

function test_instr_store()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_store!(state)
    @test state.instructionpointer == 4
    @test state.variables[6] == 3
end

function test_instr_load()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6, state)
    instr_pushval!(6, state)
    instr_pushval!(3, state)
    instr_store!(state)
    @test state.variables[6] == 3
    instr_load!(state)
    @test state.instructionpointer == 6
    @test first(state.stack) == 3
end


function test_convert_discrete_to_continuous()
    contstate = VMState(args.stackdepth, args.programlen, allvalues)
    state = DiscreteVMState()
    #instr_pushval!(6,state)
    newcontstate =
        convert_discrete_to_continuous(state, args.stackdepth, args.programlen, allvalues)
    #@test contstate == newcontstate
    @test contstate.instructionpointer == newcontstate.instructionpointer
    @test contstate.stackpointer == newcontstate.stackpointer
    @test contstate.stack == newcontstate.stack
end

function test_convert_continuous_to_discrete()
    # test true
    contstate = VMState(args.stackdepth, args.programlen, allvalues)
    discretestate = DiscreteVMState()
    #instr_pushval!(6,state)
    newdiscretestate = convert_continuous_to_discrete(
        contstate,
        args.stackdepth,
        args.programlen,
        allvalues,
    )
    #@test contstate == newcontstate
    @test discretestate.instructionpointer == newdiscretestate.instructionpointer

    @test discretestate.stack == newdiscretestate.stack
end

# TODO make instruction map cont->discrete
# run different combinations and compare.

function test_all_single_instr()
    instructions = [
        instr_pass!,
        # instr_halt!,
        # instr_pushval!,
        # instr_pop!,
        instr_dup!,
        #instr_swap!,
        instr_add!,
        instr_sub!,
        instr_mult!,
        instr_div!,
        # instr_not!,
        # instr_and!,
        # instr_goto!,
        # instr_gotoif!,
        # instr_iseq!,
        # instr_isgt!,
        # instr_isge!,
        # instr_store!,
        # instr_load!
    ]
    for instr in instructions
        test_program_conversion([instr])
    end
end

function test_program_conversion(program)
    # test true
    contstate = VMState(args.stackdepth, args.programlen, allvalues)
    discretestate = DiscreteVMState()
    # Put in some misc val (TODO randomize?)
    for val in [1, 3, 4, 2]
        contstate = instr_pushval!(val, contstate)
        instr_pushval!(val, discretestate)
    end
    for instr in program
        contstate = instr(contstate)
        instr(discretestate)
    end
    contstate = normalize_stackpointer(contstate)
    newdiscretestate = convert_continuous_to_discrete(
        contstate,
        args.stackdepth,
        args.programlen,
        allvalues,
    )
    newcontstate = convert_discrete_to_continuous(
        discretestate,
        args.stackdepth,
        args.programlen,
        allvalues,
    )

    display(contstate.stack)
    run_equality_test(contstate, newcontstate)
    run_equality_test(discretestate, newdiscretestate)
end

function test_add_probvec()
    x = [0.0, 0.0, 0.1, 0.9, 0.0]
    y = [0.0, 0.0, 0.7, 0.3, 0.0]
    result = op_probvec(+, x, y; numericvalues = [-Inf, 0, 1, Inf])
    @test sum(result) == 1.0
    @test result == [0.0, 0.0, 0.1 * 0.7, 0.1 * 0.3 + 0.7 * 0.9, 0.3 * 0.9]

    x = [0.1, 0.0, 0.1, 0.8, 0.0]
    y = [0.3, 0.0, 0.4, 0.3, 0.0]
    result = op_probvec(+, x, y; numericvalues = [-Inf, 0, 1, Inf])
    @test result == [
        0.1 * 0.3 + 0.1 * (1 - 0.3) + (1 - 0.1) * 0.3,
        0.0,
        0.1 * 0.4,
        0.1 * 0.3 + 0.4 * 0.8,
        0.3 * 0.8,
    ]
    @test sum(result) == 1.0
end

function test_div_probvec()
    x = [0.0, 0.0, 0.1, 0.9, 0.0]
    y = [0.0, 0.0, 0.7, 0.3, 0.0]
    result = op_probvec(/, x, y; numericvalues = [-Inf, 0, 1, Inf])
    @test sum(result) == 1.0
    @test result == [0.0, 0.0, 0.1 * 0.7 + 0.1 * 0.3, 0.9 * 0.3, 0.9 * 0.7]

    x = [0.05, 0.1, 0.15, 0.2, 0.5]
    y = [0.03, 0.13, 0.23, 0.33, 0.28]
    result = op_probvec(/, x, y; numericvalues = [-Inf, -1, 0, 1, Inf])
    @test sum(result) == 1.0
    # sum -Inf/0 -Inf/1 Inf/-1 -1/0
    @test result[1] == 0.05 * 0.23 + 0.05 * 0.33 + 0.28 * 0.13 + 0.1 * 0.23
end

function test_pop_vmstate()
    state = VMState(args.stackdepth, args.programlen, allvalues)
    newval = valhot(2, allvalues)
    state = push(state, newval)
    state, popval = pop(state)
    display(popval)
    @test newval == popval
end

function test_push_vmstate()
    state = VMState(args.stackdepth, args.programlen, allvalues)
    newval = valhot(2, allvalues)
    state = push(state, newval)
    @test state.stack[:, end] == newval
end

function run_equality_test(x::DiscreteVMState, y::DiscreteVMState)
    @test x.instructionpointer == y.instructionpointer
    @test x.variables == y.variables
    @test x.ishalted == y.ishalted
    @test x.stack == y.stack
end

function run_equality_test(x::VMState, y::VMState)
    @test x.instructionpointer == y.instructionpointer
    @test x.stackpointer == y.stackpointer
    @test x.stack == y.stack
end


function ==(x::DiscreteVMState, y::DiscreteVMState)
    x.instructionpointer == y.instructionpointer &&
        x.stack == y.stack &&
        x.variables == y.variables &&
        x.ishalted == y.ishalted
end

function ==(x::VMState, y::VMState)
    x.instructionpointer == y.instructionpointer &&
        x.stackpointer == y.stackpointer &&
        x.stack == y.stack
end

test_push_vmstate()
test_pop_vmstate()
test_add_probvec()
test_instr_halt()
test_instr_pushval()
test_instr_pop()
test_instr_dup()
test_instr_swap()
test_instr_add()
test_instr_sub()
test_instr_mult()
test_instr_div()
test_instr_not()
test_instr_and()
test_instr_goto()
test_instr_gotoif()
test_instr_iseq()
test_instr_isgt()
test_instr_isge()
test_instr_store()
test_instr_load()
test_convert_discrete_to_continuous()
test_convert_continuous_to_discrete()
test_all_single_instr()
