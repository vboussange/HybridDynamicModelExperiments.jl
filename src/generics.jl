abstract type AbstractOptimBackend end
struct InferICs{T} end
InferICs(b::Bool) = InferICs{b}()
istrue(::InferICs{val}) where val = val

struct LuxBackend <: AbstractOptimBackend end
struct MCMCBackend <: AbstractOptimBackend end
struct VIBackend <: AbstractOptimBackend end