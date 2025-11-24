from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import math


def _math_env() -> dict:
    """Entorno seguro con funciones matemáticas estándar."""
    env = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    env.update({"abs": abs, "pow": pow})
    return env


def parse_function(func_str: str) -> Callable[[float], float]:
    """Convierte una cadena en una función evaluable en x.

    Admite expresiones como ``"x**3 - 4*x"`` o una lambda explícita
    ``"lambda x: x**2"``. Solo expone funciones del módulo ``math``.
    """
    env = _math_env()
    cleaned = func_str.strip()

    try:
        if cleaned.startswith("lambda"):
            fn = eval(cleaned, {"__builtins__": {}}, env)
            if not callable(fn):
                raise ValueError
            return fn

        def fn(x: float) -> float:
            local_env = {**env, "x": x}
            return float(eval(cleaned, {"__builtins__": {}}, local_env))

        # Validación rápida para detectar expresiones inválidas
        fn(0.0)
        return fn
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "No se pudo interpretar la función. Usa 'x' y funciones de math (sin eval directo)."
        ) from exc


@dataclass
class RootIteration:
    iteracion: int
    a: float
    b: float
    fa: float
    fb: float
    xr: float
    fxr: float
    error_rel_pct: Optional[float]
    accion: str


class RootFinding:
    @staticmethod
    def _relative_error(current: float, previous: Optional[float]) -> Optional[float]:
        if previous is None or current == 0:
            return None
        return abs((current - previous) / current) * 100

    @staticmethod
    def _validate_interval(func: Callable[[float], float], a: float, b: float) -> Tuple[float, float]:
        fa = func(a)
        fb = func(b)
        if fa * fb >= 0:
            raise ValueError("El intervalo no es válido porque f(a)·f(b) ≥ 0.")
        return fa, fb

    @classmethod
    def biseccion(
        cls, func: Callable[[float], float], a: float, b: float, error_tol: float, max_iter: int = 50
    ) -> List[RootIteration]:
        fa, fb = cls._validate_interval(func, a, b)
        xr_prev: Optional[float] = None
        iterations: List[RootIteration] = []

        for i in range(1, max_iter + 1):
            xr = (a + b) / 2
            fxr = func(xr)
            err_pct = cls._relative_error(xr, xr_prev)

            if fxr == 0:
                accion = "Se encontró una raíz exacta en el punto medio."
            elif fa * fxr < 0:
                accion = "f(a)·f(xr) < 0 ⇒ el nuevo b es xr."
                b, fb = xr, fxr
            else:
                accion = "f(a)·f(xr) > 0 ⇒ el nuevo a es xr."
                a, fa = xr, fxr

            iterations.append(
                RootIteration(
                    iteracion=i,
                    a=a,
                    b=b,
                    fa=fa,
                    fb=fb,
                    xr=xr,
                    fxr=fxr,
                    error_rel_pct=err_pct,
                    accion=accion,
                )
            )

            xr_prev = xr
            if fxr == 0:
                break
            if err_pct is not None and err_pct / 100 < error_tol:
                break

        return iterations

    @classmethod
    def falsa_posicion(
        cls, func: Callable[[float], float], a: float, b: float, error_tol: float, max_iter: int = 50
    ) -> List[RootIteration]:
        fa, fb = cls._validate_interval(func, a, b)
        xr_prev: Optional[float] = None
        iterations: List[RootIteration] = []

        for i in range(1, max_iter + 1):
            xr = b - fb * (a - b) / (fa - fb)
            fxr = func(xr)
            err_pct = cls._relative_error(xr, xr_prev)

            if fxr == 0:
                accion = "Se encontró una raíz exacta en xr."
            elif fa * fxr < 0:
                accion = "f(a)·f(xr) < 0 ⇒ el nuevo b es xr."
                b, fb = xr, fxr
            else:
                accion = "f(a)·f(xr) > 0 ⇒ el nuevo a es xr."
                a, fa = xr, fxr

            iterations.append(
                RootIteration(
                    iteracion=i,
                    a=a,
                    b=b,
                    fa=fa,
                    fb=fb,
                    xr=xr,
                    fxr=fxr,
                    error_rel_pct=err_pct,
                    accion=accion,
                )
            )

            xr_prev = xr
            if fxr == 0:
                break
            if err_pct is not None and err_pct / 100 < error_tol:
                break

        return iterations
