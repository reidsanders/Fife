import Base: +, -, *, /, length, convert, ==, <, >
using Parameters: @with_kw
using DataStructures: CircularDeque, CircularBuffer, Deque, DefaultDict

####################
# StackValueType   #
####################
DiscreteStackFloat = Float32
@with_kw struct StackValue
    val::Int = 0
    blank::Bool = true
    max::Bool = false
    min::Bool = false
    # OrderedPair(x1,x2,x3,x4) = x > y ? error("out of order") : new(x,y)
end
# StackValue(x) = StackValue(val = x)
function StackValue(x)
    if x >= args.maxint
        return StackValue(val=0,blank=false,max=true,min=false)
    elseif x <= -args.maxint
        return StackValue(val=0,blank=false,max=false,min=true)
    else
        return StackValue(val=x,blank=false,max=false,min=false)
    end
end

function +(x::StackValue, y::StackValue)
    if x.blank || y.blank
        return StackValue()
    elseif x.max
        if y.min 
            return StackValue(0)
        else
            return StackValue(blank=false, max=true)
        end
    elseif y.max
        if x.min 
            return StackValue(0)
        else
            return StackValue(blank=false, max=true)
        end
    elseif y.min || x.min
        return StackValue(blank=false, min=true)
    end

    StackValue(x.val + y.val)
end

function *(x::StackValue, y::StackValue)
    if x.blank || y.blank
        return StackValue()
    elseif (x.max && y.min) || (x.min && y.max)
        return StackValue(blank=false, min=true)
    elseif (x.max && y.max) || (x.min && y.min)
        return StackValue(blank=false, max=true)
    elseif x.max || y.max
        return StackValue(blank=false, max=true)
    elseif x.min || y.min
        return StackValue(blank=false, min=true)
    end

    StackValue(x.val * y.val)
end

function *(x::Number, y::StackValue)
    if y.blank
        return StackValue()
    elseif (y.max && x > 0) || (y.min && x < 0)
        return StackValue(blank=false, max=true)
    elseif (y.max && x < 0) || (y.min && x > 0)
        return StackValue(blank=false, min=true)
    end

    StackValue(x * y.val)
end
x::StackValue * y::Number = y * x

# Division
function /(x::StackValue, y::StackValue)
    if x.blank || y.blank
        return StackValue()
    elseif (x.max && y.min) || (x.min && y.max)
        return StackValue(-1)
    elseif (x.max && y.max) || (x.min && y.min)
        return StackValue(1)
    elseif (x.max || x.val > 0) && y == 0
        return StackValue(blank=false, max=true)
    elseif (x.min || x.val < 0) && y == 0
        return StackValue(blank=false, min=true)
    elseif x.val == 0
        return StackValue(0)
    elseif x.min
        return StackValue(blank=false, min=true)
    elseif x.max
        return StackValue(blank=false, max=true)
    elseif y.min || y.max
        return StackValue(0)
    end

    StackValue(round(x.val / y.val))
end

function /(x::Number, y::StackValue)
    if y.blank
        return StackValue()
    elseif x > 0 && y == 0
        return StackValue(blank=false, max=true)
    elseif x < 0 && y == 0
        return StackValue(blank=false, min=true)
    elseif x == 0 || y.max || y.min
        return StackValue(0)
    end

    StackValue(round(x / y.val))
end

function /(x::StackValue, y::Number)
    if x.blank
        return StackValue()
    elseif x.max & y ≥ 0 
        return StackValue(blank=false, max=true)
    elseif x.min & y ≥ 0 
        return StackValue(blank=false, min=true)
    elseif x.max & y ≤ 0 
        return StackValue(blank=false, min=true)
    elseif x.min & y ≤ 0 
        return StackValue(blank=false, max=true)
    elseif x.val > 0 & y == 0
        return StackValue(blank=false, max=true)
    elseif x.val < 0 & y == 0
        return StackValue(blank=false, min=true)
    end

    StackValue(round(x.val / y))
end

# Comparison operators
function <(x::StackValue, y::StackValue)
    # TODO blank comparison is semantically unclear
    if x.blank || y.blank
        return false
    elseif x.max || y.min
        return false
    elseif x.min || y.max
        return true
    end

    return x.val < y.val
end
x::Number < y::StackValue = StackValue(x) < y
x::StackValue < y::Number = x < StackValue(y)


x::Number + y::StackValue = StackValue(x) + y
x::StackValue + y::Number = y + x
x::StackValue - y::StackValue = x + -1 * y
x::Number - y::StackValue = StackValue(x) - y
x::StackValue - y::Number = x - StackValue(y)


function ==(x::StackValue, y::StackValue)
    if y.blank && x.blank || y.max && x.max || y.min && x.min
        return true
    end
    x.val == y.val
end

x::StackValue == y::Number = x == StackValue(y)
x::Number == y::StackValue = StackValue(x) == y

function convert(::Type{StackValue}, x::Number)
    StackValue(x)
end

function convert(::Type{CircularDeque{T}}, x::Array{T,1}) where {T}
    y = CircularDeque{T}(length(x))
    for el in x
        push!(y, el)
    end
    y
end

function convert(::Type{CircularDeque{T}}, x::CircularDeque{StackValue}) where {T}
    y = CircularDeque{StackValue}(length(x))
    for el in x
        push!(y, el)
    end
    y
end
