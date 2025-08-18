abstract type AbstractOptimBackend end
struct InferICs{T} end
InferICs(b::Bool) = InferICs{b}()

struct LuxBackend <: AbstractOptimBackend end
struct TuringBackend <: AbstractOptimBackend end