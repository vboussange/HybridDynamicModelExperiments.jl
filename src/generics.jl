import ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries

abstract type AbstractOptimBackend end

abstract type AbstractExperimentalSetup end

struct InferICs{T, U0}
    u0_constraint::U0
end
InferICs(b::Bool, u0_constraint::U0=NoConstraint()) where U0 = InferICs{b,U0}(u0_constraint)
is_ics_estimated(::InferICs{val, U0}) where {val, U0} = val
get_u0_constraint(infer_ics::InferICs) = infer_ics.u0_constraint

@concrete struct WithValidation
    infer_ics::InferICs
    dataloader <: Union{SegmentedTimeSeries, Nothing}
end
WithValidation(infer_ics) = WithValidation(infer_ics, nothing)
is_ics_estimated(w::WithValidation) = is_ics_estimated(w.infer_ics)
get_u0_constraint(w::WithValidation) = get_u0_constraint(w.infer_ics)
