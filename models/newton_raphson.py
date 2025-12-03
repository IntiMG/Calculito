# models/newton_raphson.py
import numpy as np

def newton_raphson(f_num, df_num, x0, tol=1e-10, max_iter=50):
    xk = x0
    historia = []

    for i in range(1, max_iter + 1):
        try:
            fx = f_num(xk)
            dfx = df_num(xk)
        except:
            return {
                "error": "Error numérico",
                "mensaje": "Función no evaluable en el punto actual."
            }

        if abs(dfx) < 1e-14:
            return {
                "error": "Derivada cero",
                "mensaje": f"f'(x) ≈ 0 en x = {xk:.10f}. Método falla."
            }

        xk_new = xk - fx / dfx
        error = abs(xk_new - xk) if i > 1 else None

        historia.append({
            "n": i,
            "xk": xk,
            "f(xk)": fx,
            "f'(xk)": dfx,
            "xk+1": xk_new,
            "error": error
        })

        if error is not None and error < tol:
            f_final = f_num(xk_new)
            return {
                "raiz": xk_new,
                "f_raiz": f_final,
                "iteraciones": i,
                "historia": historia,
                "convergio": abs(f_final) < 1e-8
            }

        xk = xk_new

    return {
        "error": "Máximo de iteraciones",
        "mensaje": "No convergió en el número máximo de iteraciones.",
        "historia": historia
    }