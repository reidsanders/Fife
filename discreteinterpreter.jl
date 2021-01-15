using Pkg
Pkg.activate(".")
using Debugger
using Base
import Base: +,-,*,length
using BenchmarkTools
using ProgressMeter
using Base.Threads: @threads
using Parameters: @with_kw
using Profile
using DataStructures: CircularDeque, DefaultDict
using Test: @test

@with_kw mutable struct Args
    stackdepth::Int = 10
    programlen::Int = 5
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 3
    usegpu::Bool = false
end
args = Args()

@with_kw mutable struct DiscreteVMState
    instructionpointer::Int = 1
    stack::CircularDeque{Int} = CircularDeque{Int}(args.stackdepth)
    variables::DefaultDict{Int,Int} = DefaultDict{Int,Int}(0)
    ishalted::Bool = false
end

instr_pass(state::DiscreteVMState) = state


function instr_halt!(state::DiscreteVMState)
    state.ishalted = true
    state.instructionpointer += 1
end

function instr_pushval!(value::Integer, state::DiscreteVMState)
    push!(state.stack, value)
    state.instructionpointer += 1
end

function instr_pop!(state::DiscreteVMState)
    pop!(state.stack)
    state.instructionpointer += 1
end

function instr_dup!(state::DiscreteVMState)
    x = pop!(state.stack)
    push!(state.stack, x)
    push!(state.stack, x)
    state.instructionpointer += 1
end

function instr_swap!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    push!(state.stack, x)
    push!(state.stack, y)
    state.instructionpointer += 1
end

function instr_add!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x+y)
    state.instructionpointer += 1
end

function instr_sub!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x-y)
    state.instructionpointer += 1
end

function instr_mult!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x*y)
    state.instructionpointer += 1
end

function instr_div!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    # Floor or Round?
    push!(state.stack, floor(x/y))
    state.instructionpointer += 1
end

function instr_not!(state::DiscreteVMState)
    # 0 is false, anything else is true.
    # but if true still set to 1
    x = pop!(state.stack) 
    notx = 1 * (x != 0)
    push!(state.stack, notx)
    state.instructionpointer += 1
end

function instr_and!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    res = 1 * (x!=0 && y!=0)
    push!(state.stack, res)
    state.instructionpointer += 1
end

function instr_or!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    res = 1 * (x!=0 || y!=0)
    push!(state.stack, res)
    state.instructionpointer += 1
end

function instr_goto!(state::DiscreteVMState)
    x = pop!(state.stack)
    # Verification of non zero, positive integer?
    if x > 0
        state.instructionpointer = x
    else
        state.instructionpointer += 1
    end
end

function instr_gotoif!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x != 0 && y > 0
        state.instructionpointer = y
    else
        state.instructionpointer += 1
    end
end

function instr_iseq!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x == y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
    state.instructionpointer += 1
end

function instr_isgt!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x > y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
    state.instructionpointer += 1
end

function instr_isge!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    if x >= y
        push!(state.stack, 1)
    else
        push!(state.stack, 0)
    end
    state.instructionpointer += 1
end

function instr_store!(state::DiscreteVMState)
    x = pop!(state.stack)
    y = pop!(state.stack)
    state.variables[y] = x
    state.instructionpointer += 1
end

function instr_load!(state::DiscreteVMState)
    # TODO should this remove the address element on the stack or not
    x = pop!(state.stack)
    push!(state.stack, state.variables[x])
    state.instructionpointer += 1
end

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
