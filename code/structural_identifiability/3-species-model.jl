using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = x1(t) * ( 1 - x1(t)) - a1 * x1(t) / (b1 + x1(t)) * x2(t),
    x2'(t) = a1 * x1(t) / (b1 + x1(t)) * x2(t) - a2 * x2(t) / (b2 + x2(t)) * x3(t) - d2 * x2(t),
    x3'(t) = a2 * x2(t) / (b2 + x2(t)) * x3(t) - d3 * x3(t),
    # y1(t) = x1(t),
    # y2(t) = x2(t),
    y3(t) = x3(t)
)
id = assess_identifiability(ode)

println(id)
