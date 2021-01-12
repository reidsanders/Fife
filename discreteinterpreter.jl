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
using DataStructures: CircularDeque
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
    ishalted::Bool = false
end

instr_pass(state::DiscreteVMState) = state

function instr_pushval!(value::Integer, state::DiscreteVMState)
    push!(state.stack, value)
    state.instructionpointer += 1
end

function instr_add!(state::DiscreteVMState)
    x = pop!(state.stack) 
    y = pop!(state.stack) 
    push!(state.stack, x+y)
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

function instr_halt!(state::DiscreteVMState)
    state.ishalted = true
    state.instructionpointer += 1
end

function test_instr_pushval()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    @test state.instructionpointer == 2
    @test last(state.stack) == 3
end

function test_instr_halt()
    state = DiscreteVMState()
    instr_halt!(state)
    @test state.instructionpointer == 2
    @test state.ishalted
end

function test_instr_add()
    state = DiscreteVMState()
    instr_pushval!(3,state)
    instr_pushval!(5,state)
    instr_add!(state)
    @test state.instructionpointer == 4
    @test last(state.stack) == 8
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

test_instr_pushval()
test_instr_halt()
test_instr_add()
test_instr_mult()
test_instr_div()
