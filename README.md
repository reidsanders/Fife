# Fife

Program induction on a differentiable stack based VM. 

Idea is to allow program induction via gradient descent. This allows it to be integrated with NN based systems to merge the pattern matching capabilities of NN with the simple calculation ability of standard code.

There are two main components. 
1. A 'discrete' stack based VM with minimal concatenative instruction set.
2. A 'superposition' of the above where every state is a probability vector.

An instruction for the superposition VM is a probability vector of all instructions. Running it should produce the weighted sum of running all possible instructions simultaneously. Thus given sets of input and output pairs a satisficing input program can be learned via gradient descent.

See test/runtests.jl. 

Currently rather buggy.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reidsanders.github.io/Fife.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reidsanders.github.io/Fife.jl/dev)
[![Build Status](https://github.com/reidsanders/Fife.jl/workflows/CI/badge.svg)](https://github.com/reidsanders/Fife.jl/actions)
[![Coverage](https://codecov.io/gh/reidsanders/Fife.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/reidsanders/Fife.jl)
