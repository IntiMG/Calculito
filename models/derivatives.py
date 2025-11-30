# models/derivadas.py
from sympy import symbols, sympify, diff, lambdify, latex, N

x = symbols('x')

def procesar_funcion(expr_str: str):
    """
    Convierte una función en string a:
    - f_sym: simbólica
    - f_num(x): evaluable
    - df_num(x): derivada evaluable
    """
    try:
        expr_str = expr_str.strip().replace("^", "**")
        f_sym = sympify(expr_str, locals={'ln': 'log', 'log': 'log'})
        f_num = lambdify(x, f_sym, modules=['numpy', 'math'])
        df_sym = diff(f_sym, x)
        df_num = lambdify(x, df_sym, modules=['numpy', 'math'])

        return {
            "sym": f_sym,
            "num": f_num,
            "df_sym": df_sym,
            "df_num": df_num,
            "latex": latex(f_sym),
            "latex_df": latex(df_sym),
            "str": str(f_sym)
        }
    except Exception as e:
        raise ValueError(f"Función inválida: {e}")