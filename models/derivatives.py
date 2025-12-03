import re
from sympy import symbols, sympify, lambdify, diff, latex
from sympy.parsing.sympy_parser import parse_expr

x = symbols('x')

def procesar_funcion(expr_str: str):
    """
    Convierte un string como '5x**3 - 2x + sin(x)' → función válida de SymPy
    Acepta casi cualquier cosa que escriba el usuario
    """
    if not expr_str.strip():
        raise ValueError("Función vacía")

    original = expr_str.strip()
    s = original

    # MAGIA: Insertar * donde falte entre número/letra y x o (
    # Ejemplos:
    # 5x → 5*x
    # 2(x+1) → 2*(x+1)
    # x2 → x*2
    # pi x → pi*x

    s = re.sub(r'(\d)([x(])', r'\1*\2', s)        # 5x, 5(
    s = re.sub(r'([x\)])([0-9(])', r'\1*\2', s)   # x2, x(, )5, )(
    s = re.sub(r'(pi|E|π|e)([^\w]|$)', r'\1*', s) # pi x → pi*x
    s = re.sub(r'([\)x])([a-zA-Zπ])', r'\1*\2', s) # )sin → )*sin

    # Reemplazos comunes
    s = s.replace('^', '**')
    s = s.replace('π', 'pi')
    s = s.replace('sen', 'sin')

    try:
        # Usamos parse_expr: más tolerante que sympify
        expr = parse_expr(s, local_dict={'x': x}, transformations='all')

        f_num = lambdify(x, expr, modules=['numpy', 'math'])
        df_expr = diff(expr, x)
        df_num = lambdify(x, df_expr, modules=['numpy', 'math'])

        return {
            "sym": expr,
            "num": f_num,
            "df_sym": df_expr,
            "df_num": df_num,
            "latex": latex(expr),
            "str": str(expr)
        }
    except Exception as e:
        raise ValueError(f"Función no válida: {original}\nCorrección intentada: {s}\nError: {e}")