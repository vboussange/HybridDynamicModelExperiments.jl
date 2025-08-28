abstract type AbstractOptimBackend end
struct InferICs{T} end
InferICs(b::Bool) = InferICs{b}()
istrue(::InferICs{val}) where val = val

struct MCMCBackend <: AbstractOptimBackend end
struct VIBackend <: AbstractOptimBackend end