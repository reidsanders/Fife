function partial(f, a...)
    ((b...) -> f(a..., b...))
end

function replacenans(x, replacement)
    if isnan(x)
        return replacement
    else
        return x
    end
end

function setoutofboundstolarge(x::Number; min = -Inf, max = Inf, large = Inf)::Number
    if x > max
        return large
    elseif x < min
        return -large
    else
        return x
    end
end

function roundnoninf(x::Number)
    if x in [Inf, -Inf]
        return x
    end
    round(Int, x)
end

function printall(x::AbstractArray)
    for row in eachrow(x)
        println(row)
    end
end
