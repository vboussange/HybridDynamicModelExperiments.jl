using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -r1 * ( x1(t) * ( 1 + b1 * x1(t) + alpha *  (x2(t) + x3(t))) + mu * ( 2 * x1(t) - x2(t) - x3(t)) + delta * (u1(t) - x1(t))),
    x2'(t) = -r2 * ( x2(t) * ( 1 + b2 * x2(t) + alpha *  (x1(t) + x3(t))) + mu * ( 2 * x2(t) - x1(t) - x3(t)) + delta * (u2(t) - x2(t))),
    x3'(t) = -r3 * ( x3(t) * ( 1 + b3 * x3(t) + alpha *  (x1(t) + x2(t))) + mu * ( 2 * x3(t) - x2(t) - x1(t)) + delta * (u3(t) - x3(t))),
    y1(t) = x1(t),
    y2(t) = x2(t),
    y3(t) = x3(t)
)
id = assess_identifiability(ode)

println(id)