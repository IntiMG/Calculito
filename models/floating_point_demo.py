# models/floating_point_demo.py

"""
Módulo: floating_point_demo
----------------------------------------------

Objetivo: ilustrar con ejemplos que la computadora NO representa
exactamente todos los números reales, usando el caso clásico:

    print(0.1 + 0.2 == 0.3)

y reforzándolo con NumPy:

    np.isclose(0.1 + 0.2, 0.3)

COMENTARIOS TEÓRICOS (para el profe):

- Los números se almacenan en formato IEEE 754 (por ejemplo float64),
  que usa una cantidad FINITA de bits para representar valores reales.
- Fracciones como 0.1 o 0.2 son finitas en base 10, pero en base 2 se
  vuelven representaciones periódicas (como 1/3 = 0.3333... en base 10).
- Por eso 0.1, 0.2 y 0.3 no se guardan exactamente, sino como
  aproximaciones MUY cercanas en binario.
- Cuando comparamos con '==' estamos pidiendo igualdad EXACTA bit a bit.
- En cómputo científico se usa una comparación con tolerancia:

      np.isclose(a, b, rtol, atol)

  donde rtol es la tolerancia relativa y atol la absoluta.

Además comparamos:
- float32 vs float64
- epsilon de máquina: np.finfo(tipo).eps
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FloatAnalysis:
    """
    Información detallada sobre la suma a + b y su comparación con c.

    Campos principales:
        a, b, c          → valores de entrada (float)
        sum_ab           → resultado de a + b
        equal            → resultado de (a + b == c) con float estándar
        diff             → (a + b) - c  (error absoluto)

    Campos de análisis extendido con NumPy:
        isclose          → np.isclose(a + b, c) con tolerancia estándar
        rtol, atol       → tolerancias usadas en isclose
        eps_float64      → epsilon de máquina para float64
        eps_float32      → epsilon de máquina para float32
        sum32, sum64     → suma en precisión float32 y float64
        sum32_repr       → suma float32 con 20 decimales
        sum64_repr       → suma float64 con 20 decimales

    Campos de representación:
        a_repr, b_repr, c_repr, sum_repr → números en formato .20f
    """
    a: float
    b: float
    c: float
    sum_ab: float
    equal: bool
    diff: float

    isclose: bool
    rtol: float
    atol: float
    eps_float64: float
    eps_float32: float
    sum32: float
    sum64: float
    sum32_repr: str
    sum64_repr: str

    a_repr: str
    b_repr: str
    c_repr: str
    sum_repr: str


class FloatingPointDemo:

    @staticmethod
    def analyze(a: float, b: float, c: float,
                rtol: float = 1e-09,
                atol: float = 0.0) -> FloatAnalysis:
        """
        Calcula a + b, lo compara con c y devuelve un objeto FloatAnalysis.

        - sum_ab      → a + b (float estándar de Python ≈ float64)
        - equal       → (a + b == c)
        - diff        → sum_ab - c
        - isclose     → np.isclose(sum_ab, c, rtol, atol)
        - eps_float64 → eps de máquina para float64
        - eps_float32 → eps de máquina para float32
        - sum32       → suma usando np.float32
        - sum64       → suma usando np.float64 (equivalente a float)
        """
        # Suma en precisión "normal" (float de Python)
        sum_ab = a + b
        equal = (sum_ab == c)
        diff = sum_ab - c

        # Comparación con tolerancia (forma correcta en cómputo científico)
        isclose = bool(np.isclose(sum_ab, c, rtol=rtol, atol=atol))

        # Epsilon de máquina: distancia entre 1.0 y el siguiente número representable
        eps64 = np.finfo(np.float64).eps
        eps32 = np.finfo(np.float32).eps

        # Misma operación pero forzando precisión float32 y float64
        a32 = np.float32(a)
        b32 = np.float32(b)
        c32 = np.float32(c)
        sum32 = float(a32 + b32)  # convertimos a float para imprimir fácil

        a64 = np.float64(a)
        b64 = np.float64(b)
        c64 = np.float64(c)
        sum64 = float(a64 + b64)

        return FloatAnalysis(
            a=a,
            b=b,
            c=c,
            sum_ab=sum_ab,
            equal=equal,
            diff=diff,
            isclose=isclose,
            rtol=rtol,
            atol=atol,
            eps_float64=eps64,
            eps_float32=eps32,
            sum32=sum32,
            sum64=sum64,
            sum32_repr=format(sum32, ".20f"),
            sum64_repr=format(sum64, ".20f"),
            a_repr=format(a, ".20f"),
            b_repr=format(b, ".20f"),
            c_repr=format(c, ".20f"),
            sum_repr=format(sum_ab, ".20f"),
        )
