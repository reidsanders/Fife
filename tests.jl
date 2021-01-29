using Test
using Flux
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using DataStructures: CircularDeque, DefaultDict
using Flux: onehot, onehotbatch, onecold, crossentropy, logitcrossentropy, glorot_uniform, mse, epseltype
using CUDA 
using Random

include("main.jl")

include("utils.jl")
using .Utils: partial

include("discreteinterpreter.jl")
using .DiscreteInterpreter
using .SuperInterpreter

######################################
# Global initialization
######################################
@with_kw mutable struct Args
    batchsize::Int = 2
    lr::Float32 = 2e-4
    epochs::Int = 2
    stackdepth::Int = 10
    programlen::Int = 5
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 3
    usegpu::Bool = false
end
include("parameters.jl")

function init_random_state(stackdepth, programlen, allvalues)
    stack = rand(Float32, length(allvalues), stackdepth)
    instructionpointer = zeros(Float32, programlen, )
    stackpointer = zeros(Float32, stackdepth, )
    #instructionpointer = rand(Float32, programlen, )
    #stackpointer = rand(Float32, stackdepth, )
    stack = normit(stack)
    instructionpointer[1] = 1.f0
    stackpointer[1] = 1.f0
    state = VMState(
        instructionpointer |> device,
        stackpointer |> device,
        stack |> device,
    )
    #normit(state)
end

allvalues = [i for i in 0:5]
instr_2 = partial(instr_val, valhot(2, allvalues))

blank_state = VMState(3, 4, allvalues)
blank_state_random = init_random_state(3, 4, allvalues)
check_state_asserts(blank_state)
check_state_asserts(blank_state_random)

newstate = instr_dup(blank_state_random)
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
    instr_pushval!(3,state)
    @test state.instructionpointer == 2
    @test last(state.stack) == 3
end

function test_instr_pop()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_pop!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 3
end

function test_instr_dup()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_dup!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 5
    pop!(state.stack)
    @test last(state.stack) == 5
end

function test_instr_swap()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_swap!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 3
    pop!(state.stack)
    @test last(state.stack) == 5
end

function test_instr_add()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_add!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 8
end

function test_instr_sub()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_sub!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 2
end

function test_instr_mult()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_mult!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 15
end

function test_instr_div()
    state = DiscreteVMState()
    instr_pushval!(4,state)
    instr_pushval!(9,state)
    instr_div!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 2
end

function test_instr_not()
    state = DiscreteVMState()
    # test true
    instr_pushval!(3,state)
    instr_not!(state)
    @test state.instructionpointer == 3
    @test last(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(0,state)
    instr_not!(state)
    @test state.instructionpointer == 3
    @test last(state.stack) == 0
end

function test_instr_and()
    state = DiscreteVMState()
    # test true
    instr_pushval!(3,state)
    instr_pushval!(1,state)
    instr_and!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(0,state)
    instr_and!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 0
end

function test_instr_goto()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_goto!(state)
    @test state.instructionpointer == 6
    # test false
    state = DiscreteVMState()
    instr_pushval!(-1,state)
    instr_goto!(state)
    @test state.instructionpointer == 3
end

function test_instr_gotoif()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(3,state)
    instr_gotoif!(state)
    @test state.instructionpointer == 6
    # test false
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(0,state)
    instr_gotoif!(state)
    @test state.instructionpointer == 4
end

function test_instr_iseq()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(6,state)
    instr_iseq!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 1
    # test false
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(3,state)
    instr_iseq!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 0
end

function test_instr_isgt()
    # test false
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(6,state)
    instr_isgt!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 0
    # test true
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(6,state)
    instr_isgt!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 1
end

function test_instr_isge()
    # test false
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(3,state)
    instr_isge!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 0
    # test false
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(6,state)
    instr_isge!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 1
    # test true
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(6,state)
    instr_isge!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 1
end

function test_instr_store()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(3,state)
    instr_store!(state)
    @test state.instructionpointer == 4
    @test state.variables[6] == 3
end

function test_instr_load()
    # test true
    state = DiscreteVMState()
    instr_pushval!(6,state)
    instr_pushval!(6,state)
    instr_pushval!(3,state)
    instr_store!(state)
    @test state.variables[6] == 3
    instr_load!(state)
    @test state.instructionpointer == 6
    @test last(state.stack) == 3
end


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


function test_convert_discrete_to_continuous()
    contstate = VMState(args.stackdepth, args.programlen, allvalues)
    state = DiscreteVMState()
    #instr_pushval!(6,state)
    newcontstate = convert_discrete_to_continuous(state, args.stackdepth, args.programlen, allvalues)
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
    newdiscretestate = convert_continuous_to_discrete(contstate, args.stackdepth, args.programlen, allvalues)
    #@test contstate == newcontstate
    @test discretestate.instructionpointer == newdiscretestate.instructionpointer
    @test dicretestate.stackpointer == newdiscretestate.stackpointer
    @test discretestate.stack == newdiscretestate.stack
end

# TODO make instruction map cont->discrete
# run different combinations and compare.

test_convert_discrete_to_continuous()
test_convert_continuous_to_discrete()