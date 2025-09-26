import ConcreteStructs: @concrete
import HybridDynamicModels: AbstractSetup

@concrete struct WithValidation <: AbstractSetup
    infer_ics::InferICs
    dataloader <: Union{SegmentedTimeSeries, Nothing}
end
WithValidation(infer_ics) = WithValidation(infer_ics, nothing)
is_ics_estimated(w::WithValidation) = is_ics_estimated(w.infer_ics)
get_u0_constraint(w::WithValidation) = get_u0_constraint(w.infer_ics)
