abstract type AbstractOptimBackend end


struct InferICs{T, U0}
    u0_constraint::U0
end
InferICs(b::Bool, u0_constraint::U0=NoConstraint()) where U0 = InferICs{b,U0}(u0_constraint)
istrue(::InferICs{val, U0}) where {val, U0} = val
