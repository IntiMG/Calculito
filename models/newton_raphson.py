# models/newton_raphson.py
import numpy as np

def newton_raphson(f_num, df_num, x0, tol=1e-8, max_iter=50):
    xk = x0
    historia = []
    
    for i in range(max_iter):
        fx = f_num(xk)
        dfx = df_num(xk)
        
        if abs(dfx) < 1e-14:
            return {
                "error": "Derivada cero",
                "mensaje": f"f'(x) ≈ 0 en x = {xk:.10f}. Método falla.",
                "historia": historia
            }
        
        xk_new = xk - fx / dfx
        ea = abs(xk_new - xk) if i > 0 else None
        
        historia.append({
            "n": i+1,
            "xk": xk,
            "f(xk)": fx,
            "f'(xk)": dfx,
            "xk+1": xk_new,
            "error": ea
        })
        
        if ea is not None and ea < tol:
            f_raiz = f_num(xk_new)
            return {
                "raiz": xk_new,
                "f_raiz": f_raiz,
                "iteraciones": i+1,
                "historia": historia,
                "convergio": abs(f_raiz) < 1e-6
            }
        
        xk = xk_new
    
    return {
        "error": "Máximo de iteraciones alcanzado",
        "mensaje": "No convergió en el número máximo de iteraciones.",
        "historia": historia
    }