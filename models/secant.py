# models/secante.py
import numpy as np

def secante(f_num, x0, x1, tol=1e-8, max_iter=50):
    x_prev, x_curr = x0, x1
    historia = []
    
    for i in range(max_iter):
        f_prev = f_num(x_prev)
        f_curr = f_num(x_curr)
        
        if abs(f_curr - f_prev) < 1e-14:
            return {
                "error": "División por diferencia casi cero",
                "mensaje": "f(x_k) ≈ f(x_{k-1}) → método falla.",
                "historia": historia
            }
        
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        ea = abs(x_next - x_curr)
        
        historia.append({
            "n": i+1,
            "x_{n-1}": x_prev,
            "x_n": x_curr,
            "f(x_n)": f_curr,
            "x_{n+1}": x_next,
            "error": ea if i > 0 else None
        })
        
        if ea < tol:
            f_raiz = f_num(x_next)
            return {
                "raiz": x_next,
                "f_raiz": f_raiz,
                "iteraciones": i+1,
                "historia": historia,
                "convergio": abs(f_raiz) < 1e-6
            }
        
        x_prev, x_curr = x_curr, x_next
    
    return {
        "error": "Máximo de iteraciones",
        "mensaje": "No convergió.",
        "historia": historia
    }