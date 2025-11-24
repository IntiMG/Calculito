from dataclasses import dataclass, field
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
    error_abs: Optional[float]
    intervalo_longitud: float
    accion: str
    xr_formula: str = ""
    producto: Optional[float] = None
    intervalo_siguiente: Optional[Tuple[float, float]] = None
    detalles: List[str] = field(default_factory=list)


class RootFinding:
    @staticmethod
    def _relative_error(current: float, previous: Optional[float]) -> Optional[float]:
        if previous is None or current == 0:
            return None
        return abs((current - previous) / current) * 100

    @staticmethod
    def _absolute_error(current: float, previous: Optional[float]) -> Optional[float]:
        if previous is None:
            return None
        return abs(current - previous)

    @staticmethod
    def _fmt(x: float) -> str:
        if x is None:
            return "-"
        if x == 0:
            return "0"
        ax = abs(x)
        if ax >= 1e6 or (0 < ax < 1e-4):
            return f"{x:.6e}".rstrip("0").rstrip(".")
        return f"{x:.6f}".rstrip("0").rstrip(".")

    @classmethod
    def _build_details_biseccion(
        cls,
        a: float,
        b: float,
        fa: float,
        fb: float,
        xr: float,
        fxr: float,
        next_interval: Tuple[float, float],
    ) -> List[str]:
        fmt = cls._fmt
        next_a, next_b = next_interval
        product = fa * fxr
        if product == 0:
            sign_hint = "f(xr) = 0 -> raíz exacta en xr"
        elif product < 0:
            sign_hint = "f(a)*f(xr) < 0 -> la raíz queda en [a, xr]"
        else:
            sign_hint = "f(a)*f(xr) > 0 -> la raíz queda en [xr, b]"
        return [
            f"x_l = {fmt(a)} , x_u = {fmt(b)}",
            f"x_r = (x_l + x_u)/2 = ({fmt(a)} + {fmt(b)})/2 = {fmt(xr)}",
            f"f(x_l) = f({fmt(a)}) = {fmt(fa)}",
            f"f(x_u) = f({fmt(b)}) = {fmt(fb)}",
            f"f(x_r) = f({fmt(xr)}) = {fmt(fxr)}",
            f"f(x_l)*f(x_r) = {fmt(product)} -> intervalo siguiente: [{fmt(next_a)}, {fmt(next_b)}] ({sign_hint})",
        ]

    @classmethod
    def _build_details_falsa_posicion(
        cls,
        a: float,
        b: float,
        fa: float,
        fb: float,
        xr: float,
        fxr: float,
        next_interval: Tuple[float, float],
    ) -> List[str]:
        fmt = cls._fmt
        next_a, next_b = next_interval
        product = fa * fxr
        if product == 0:
            sign_hint = "f(xr) = 0 -> raíz exacta en xr"
        elif product < 0:
            sign_hint = "f(a)*f(xr) < 0 -> la raíz queda en [a, xr]"
        else:
            sign_hint = "f(a)*f(xr) > 0 -> la raíz queda en [xr, b]"
        return [
            f"x_l = {fmt(a)} , x_u = {fmt(b)}",
            f"x_r = b - f(b)*(a - b)/(f(a) - f(b)) = {fmt(b)} - {fmt(fb)}*({fmt(a)} - {fmt(b)})/({fmt(fa)} - {fmt(fb)}) = {fmt(xr)}",
            f"f(x_l) = f({fmt(a)}) = {fmt(fa)}",
            f"f(x_u) = f({fmt(b)}) = {fmt(fb)}",
            f"f(x_r) = f({fmt(xr)}) = {fmt(fxr)}",
            f"f(x_l)*f(x_r) = {fmt(product)} -> intervalo siguiente: [{fmt(next_a)}, {fmt(next_b)}] ({sign_hint})",
        ]

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
            err_abs = cls._absolute_error(xr, xr_prev)
            interval_len = abs(b - a)
            product = fa * fxr

            if fxr == 0:
                accion = "Se encontró una raíz exacta en xr."
                next_a, next_b = xr, xr
                next_fa, next_fb = fxr, fxr
            elif product < 0:
                accion = "f(a)*f(xr) < 0 -> el nuevo b es xr."
                next_a, next_b = a, xr
                next_fa, next_fb = fa, fxr
            else:
                accion = "f(a)*f(xr) > 0 -> el nuevo a es xr."
                next_a, next_b = xr, b
                next_fa, next_fb = fxr, fb

            detalles = cls._build_details_biseccion(a, b, fa, fb, xr, fxr, (next_a, next_b))

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
                    error_abs=err_abs,
                    intervalo_longitud=interval_len,
                    accion=accion,
                    xr_formula=f"(a + b)/2 = ({cls._fmt(a)} + {cls._fmt(b)})/2 = {cls._fmt(xr)}",
                    producto=product,
                    intervalo_siguiente=(next_a, next_b),
                    detalles=detalles,
                )
            )

            xr_prev = xr
            if fxr == 0:
                break
            if err_pct is not None and err_pct / 100 < error_tol:
                break

            a, b, fa, fb = next_a, next_b, next_fa, next_fb

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
            err_abs = cls._absolute_error(xr, xr_prev)
            interval_len = abs(b - a)
            product = fa * fxr

            if fxr == 0:
                accion = "Se encontró una raíz exacta en xr."
                next_a, next_b = xr, xr
                next_fa, next_fb = fxr, fxr
            elif product < 0:
                accion = "f(a)*f(xr) < 0 -> el nuevo b es xr."
                next_a, next_b = a, xr
                next_fa, next_fb = fa, fxr
            else:
                accion = "f(a)*f(xr) > 0 -> el nuevo a es xr."
                next_a, next_b = xr, b
                next_fa, next_fb = fxr, fb

            detalles = cls._build_details_falsa_posicion(a, b, fa, fb, xr, fxr, (next_a, next_b))

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
                    error_abs=err_abs,
                    intervalo_longitud=interval_len,
                    accion=accion,
                    xr_formula=f"xr = b - f(b)*(a - b)/(f(a) - f(b)) = {cls._fmt(b)} - {cls._fmt(fb)}*({cls._fmt(a)} - {cls._fmt(b)})/({cls._fmt(fa)} - {cls._fmt(fb)}) = {cls._fmt(xr)}",
                    producto=product,
                    intervalo_siguiente=(next_a, next_b),
                    detalles=detalles,
                )
            )

            xr_prev = xr
            if fxr == 0:
                break
            if err_pct is not None and err_pct / 100 < error_tol:
                break

            a, b, fa, fb = next_a, next_b, next_fa, next_fb

        return iterations
