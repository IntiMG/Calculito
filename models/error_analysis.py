# models/error_analysis.py
import math
<<<<<<< HEAD
#aqui da error esto
#from sympy import symbols, sympify, diff
=======
from sympy import symbols, sympify, diff
>>>>>>> 50477990be3153c1a50ac6c4f3f2ce9a6c9a2f70

class ErrorAnalysis:

    @staticmethod
    def direct_errors(x_true: float, x_approx: float):
        """
        Devuelve:
          Ea  = error absoluto
          Er  = error relativo (o None si x_true = 0)
          Er% = error relativo en porcentaje (o None si x_true = 0)
        """
        Ea = abs(x_true - x_approx)
        Er = None
        Er_pct = None

        if x_true != 0:
            Er = Ea / abs(x_true)
            Er_pct = Er * 100.0

        return Ea, Er, Er_pct

    @staticmethod
    def propagation(func_str: str, x: float, dx: float):
        """
        Propagación del error para una función f(x):

        - func_str: string con la función, ej. "sin(x) + x**2"
        - x: valor medido
        - dx: incertidumbre en x

        Devuelve un dict con:
            f_x, fpx, dy_aprox, dy_real, abs_error, rel_error, x, dx
        """
        # Variable simbólica
        x_sym = symbols("x")

        # f(x) y f'(x) simbólicos
        f_sym = sympify(func_str)
        df_sym = diff(f_sym, x_sym)

        # Evaluaciones numéricas (float)
        f_x = float(f_sym.subs(x_sym, x))
        fpx = float(df_sym.subs(x_sym, x))

        dy_aprox = fpx * dx                 # Δy ≈ f'(x)·Δx

        f_x_dx = float(f_sym.subs(x_sym, x + dx))
        dy_real = f_x_dx - f_x              # Δy real

        abs_error = abs(dy_real - dy_aprox)
        rel_error = abs_error / abs(dy_real) if dy_real != 0 else None

        return {
            "function": func_str,                 # ← ESTO ES NECESARIO
            "derivative": str(df_sym), 
            "f_x": f_x,
            "fpx": fpx,
            "dy_aprox": dy_aprox,
            "dy_real": dy_real,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "x": x,
            "dx": dx,
        }
