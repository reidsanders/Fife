module Utils
function partial(f, a...)
    return ((b...) -> f(a..., b...))
end

function replacenans(x, replacement)
    if isnan(x)
        return replacement
    else
        return x
    end
end

function setoutofboundstoinf(x::Number; min = -Inf, max = Inf)::Number
    if x < min
        return -Inf
    elseif x > max
        return Inf
    else
        return x
    end
end

function roundnoninf(x::Number)
    if x in [Inf, -Inf]
        return x
    end
    return round(Int, x)
end

function coercetostackvalue(x::Number; min = -Inf, max = Inf)
    x = replacenans(x, 0)
    x = setoutofboundstoinf(x; min = min, max = max)
    return roundnoninf(x)
end

export partial, replacenans, setoutofboundstoinf, roundnoninf, coercetostackvalue
end
