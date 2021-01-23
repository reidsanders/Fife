# TODO use modules and export?
module Utils
function partial(f, a...)
    ( (b...) -> f(a..., b...) )
end

export partial
end