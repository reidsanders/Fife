# TODO use modules and export?
function partial(f, a...)
    ( (b...) -> f(a..., b...) )
end
