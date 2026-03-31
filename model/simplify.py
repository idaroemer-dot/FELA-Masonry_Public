
import sympy as sp

fcs, fce, ls, ms, sx = sp.symbols('fcs fce ls ms sx', real=True)
xi = 1.0
A = sp.Rational(1,2)*fcs*xi*ms + sp.Rational(1,2)*fce*(1-xi)
B = sp.Rational(1,2)*fcs*xi*ls + sp.Rational(1,2)*fce*(1-xi)

sin_a = (sx + A)/B
cos_a = sp.sqrt(1 - sin_a**2)

tau = (
    sp.Rational(1,2)*fcs*((ls - ms*sin_a)/cos_a)*xi
    + sp.Rational(1,2)*fce*((1 - sin_a)/cos_a)*(1-xi)
    - sx*(sin_a/cos_a)
)

tau_simplified = sp.simplify(sp.factor(tau))
print(tau_simplified)

candidate = sp.sqrt(B**2 - (sx + A)**2)
print(sp.simplify(candidate**2))
