using Fife
using Fife: 
    valhot,
    pushtostack,
    popfromstack,
    op_probvec,
    normalize_stackpointer,
    create_random_discrete_program,
    create_trainable_mask,
    super_step,
    softmaxmask,
    applyfullmask
using Test
using Random
using Flux:
    onehot,
    onehotbatch,
    glorot_uniform
using CUDA
Random.seed!(123);
CUDA.allowscalar(false)
using Parameters: @with_kw



# args = Args(
#     batchsize = 2,
#     lr = 2e-4,
#     epochs = 2,
#     stackdepth = 8,
#     programlen = 7,
#     inputlen = 2,
#     max_ticks = 5,
#     maxint = 8,
#     trainsetsize = 10,
#     usegpu = false,
#     StackFloatType = Float32,
#     StackValueType = Int
# )
@with_kw mutable struct Args
    batchsize::Int = 2
    lr::Float32 = 2e-4
    epochs::Int = 2
    stackdepth::Int = 10
    programlen::Int = 10
    inputlen::Int = 2 # frozen part, assumed at front for now
    max_ticks::Int = 5
    maxint::Int = 20
    trainsetsize::Int = 10
    usegpu::Bool = false
    StackFloatType::Type = Float32
    StackValueType::Type = Int

    ## TODO initialization function inside Args / or inside Fife module (then export inside args?)
end
args = Args()

# @with_kw mutable struct TestArgs
#     batchsize::Int = 2
#     lr::Float32 = 2e-4
#     epochs::Int = 2
#     stackdepth::Int = 8
#     programlen::Int = 7
#     inputlen::Int = 2 # frozen part, assumed at front for now
#     max_ticks::Int = 5
#     maxint::Int = 8
#     trainsetsize::Int = 10
#     usegpu::Bool = false
#     StackFloatType::Type = Float32
#     StackValueType::Type = Int
# end
# args = TestArgs()
device,
largevalue,
coercetostackvaluepart,
numericvalues,
nonnumericvalues,
allvalues,
ishaltedvalues,
blanks,
blankstack = create_dependent_values(args)
# include("parameters.jl")
@testset "Fife.jl" begin

    ######################################
    # Global initialization
    ######################################

    function init_random_state(
        stackdepth::Int = args.stackdepth,
        programlen::Int = args.programlen,
        allvalues::Union{Array,CuArray} = allvalues,
    )
        instrpointer = zeros(Float32, programlen)
        stackpointer = zeros(Float32, stackdepth)
        ishalted = zeros(Float32, 2)
        stack = rand(Float32, length(allvalues), stackdepth)
        variables = rand(Float32, length(allvalues), stackdepth)
        stack = normit(stack)
        variables = normit(variables)
        instrpointer[1] = 1.0
        stackpointer[1] = 1.0
        ishalted[1] = 1.0 # set false
        VMState(
            instrpointer |> device,
            stackpointer |> device,
            stack |> device,
            variables |> device,
            ishalted |> device,
        )
    end

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
        @test state.instrpointer == 2
        @test state.ishalted
    end

    function test_instr_pushval()
        state = DiscreteVMState()
        instr_pushval!(3, state)
        @test state.instrpointer == 2
        @test first(state.stack) == 3
    end

    function test_instr_pop()
        state = DiscreteVMState()
        instr_pushval!(3, state)
        instr_pushval!(5, state)
        instr_pop!(state)
        @test state.instrpointer == 4
        @test first(state.stack) == 3
    end

    function test_instr_dup()
        state = DiscreteVMState()
        instr_pushval!(3, state)
        instr_pushval!(5, state)
        instr_dup!(state)
        @test state.instrpointer == 4
        @test first(state.stack) == 5
        popfirst!(state.stack)
        @test first(state.stack) == 5
    end

    function test_instr_swap()
        state = DiscreteVMState()
        instr_pushval!(3, state)
        instr_pushval!(5, state)
        instr_swap!(state)
        @test state.instrpointer == 4
        @test first(state.stack) == 3
        popfirst!(state.stack)
        @test first(state.stack) == 5
    end

    function test_instr_add()
        state = DiscreteVMState()
        instr_pushval!(3, state)
        instr_pushval!(5, state)
        instr_add!(state)
        @test state.instrpointer == 4
        @test first(state.stack) == 8
    end

    function test_instr_sub()
        state = DiscreteVMState()
        instr_pushval!(3, state)
        instr_pushval!(5, state)
        instr_sub!(state)
        @test state.instrpointer == 4
        @test first(state.stack) == 2
    end

    function test_instr_mult()
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

    function test_instr_div()
        state = DiscreteVMState()
        instr_pushval!(4, state)
        instr_pushval!(7, state)
        instr_div!(state)
        @test state.instrpointer == 4
        @test first(state.stack) == 2
    end

    function test_instr_not()
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

    function test_instr_and()
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

    function test_instr_goto()
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

    function test_instr_gotoif()
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

    function test_instr_iseq()
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

    function test_instr_isgt()
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

    function test_instr_isge()
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

    function test_instr_store()
        # test true
        state = DiscreteVMState()
        instr_pushval!(6, state)
        instr_pushval!(3, state)
        instr_store!(state)
        @test state.instrpointer == 4
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
        @test state.instrpointer == 6
        @test first(state.stack) == 3
    end


    function test_convert_discrete_to_continuous()
        contstate = VMState(args.stackdepth, args.programlen, allvalues)
        state = DiscreteVMState()
        #instr_pushval!(6,state)
        newcontstate = convert_discrete_to_continuous(state, allvalues)
        #@test contstate == newcontstate
        @test contstate.instrpointer == newcontstate.instrpointer
        @test contstate.stackpointer == newcontstate.stackpointer
        @test contstate.stack == newcontstate.stack
        return
    end

    function test_convert_continuous_to_discrete()
        # test true
        contstate = VMState(args.stackdepth, args.programlen, allvalues)
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


    function test_add_probvec()
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

    function test_div_probvec()
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

    function test_pop_vmstate()
        state = VMState(args.stackdepth, args.programlen, allvalues)
        newval = valhot(2, allvalues)
        newstate = pushtostack(state, newval)
        newstate, popval = popfromstack(newstate)
        @test newval == popval
        @test newstate.stack == state.stack
    end

    function test_push_vmstate()
        state = VMState(args.stackdepth, args.programlen, allvalues)
        newval = valhot(2, allvalues)
        state = pushtostack(state, newval)
        @test state.stack[:, end] == newval
    end

    function run_equality_test(x::DiscreteVMState, y::DiscreteVMState)
        @test x.instrpointer == y.instrpointer
        @test x.variables == y.variables
        @test x.ishalted == y.ishalted
        @test x.stack == y.stack
    end

    function run_equality_test(x::VMState, y::VMState)
        @test x.instrpointer ≈ y.instrpointer
        @test x.stackpointer ≈ y.stackpointer
        @test x.stack ≈ y.stack
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

    function test_all_single_instr()
        instructions = [
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
        ]
        for instr in instructions
            test_program_conversion([instr])
        end
    end

    function test_program_conversion(program)
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
            contstate = VMState(args.stackdepth, args.programlen, allvalues)
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
            newdiscretestate = convert_continuous_to_discrete(contstate, allvalues)
            newcontstate = convert_discrete_to_continuous(discretestate, allvalues)
            run_equality_asserts(contstate, newcontstate)
            run_equality_asserts(discretestate, newdiscretestate)
            run_equality_test(contstate, newcontstate)
            run_equality_test(discretestate, newdiscretestate)
        end
    end


    function test_super_step()
        ### TODO test super_step / run.
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
            ]
            val_instructions
        ]
        discrete_program = create_random_discrete_program(args.programlen, instructions)
        program = convert(Array{Float32}, onehotbatch(discrete_program, instructions))
        rand!(program)
        program = normit(program)
        state = VMState(args.stackdepth, args.programlen, allvalues)
        state = super_step(state, program, instructions)
        check_state_asserts(state)
    end

    function test_super_run_program()
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
        softmaxprog = partial(softmaxmask, trainmaskfull |> device)
        applyfullmaskprog = partial(applyfullmask, trainmaskfull)
        applyfullmasktohidden = partial((mask, prog) -> mask .* prog, trainmaskfull)

        hiddenprogram = hiddenprogram |> device
        program = softmaxprog(hiddenprogram) |> device
        target_program = target_program |> device
        hiddenprogram = hiddenprogram |> device
        trainmask = trainmask |> device

        blank_state = VMState(args.stackdepth, args.programlen, allvalues)
        check_state_asserts(blank_state)
        target = runprogram(blank_state, target_program, instructions, 1000)
    end

    test_push_vmstate()
    test_pop_vmstate()
    test_div_probvec()
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
    test_super_step()
    test_super_run_program()
end
