# models/numpy_workshop.py
import numpy as np


class NumpyWorkshop:
    """
    Taller guiado para mostrar:
    - Pérdida de precisión en operaciones numéricas
    - Cómo NumPy maneja números flotantes (float32 vs float64)
    """

    @staticmethod
    def _exp_sum_0_1(n):
        """Suma 0.1 n veces y compara con el valor exacto n*0.1."""
        exact = 0.1 * n

        # Python (float estándar, similar a float64)
        py_val = sum([0.1] * n)

        # NumPy float32
        v32 = np.float32(0.1)
        for _ in range(n - 1):
            v32 += np.float32(0.1)
        np32_val = float(v32)

        # NumPy float64
        v64 = np.float64(0.1)
        for _ in range(n - 1):
            v64 += np.float64(0.1)
        np64_val = float(v64)

        return {
            "title": f"Suma repetida de 0.1 ({n} veces)",
            "description": (
                "Se suma el número 0.1 muchas veces. "
                "En aritmética exacta el resultado debería ser 0.1 × n, "
                "pero en punto flotante aparecen errores de redondeo."
            ),
            "exact": exact,
            "python": {
                "value": py_val,
                "abs_error": abs(py_val - exact),
            },
            "np32": {
                "value": np32_val,
                "abs_error": abs(np32_val - exact),
            },
            "np64": {
                "value": np64_val,
                "abs_error": abs(np64_val - exact),
            },
        }

    @staticmethod
    def _exp_cancellation():
        """(1e8 + 1) - 1e8 -> cancelación de cifras significativas."""
        a = 1e8
        b = 1.0
        exact = 1.0

        py_val = (a + b) - a
        np32_val = float((np.float32(a) + np.float32(b)) - np.float32(a))
        np64_val = float((np.float64(a) + np.float64(b)) - np.float64(a))

        return {
            "title": "(1e8 + 1) - 1e8 (cancelación)",
            "description": (
                "Al sumar un número muy grande con uno muy pequeño, "
                "las cifras del pequeño pueden perderse al redondear. "
                "Debería dar exactamente 1."
            ),
            "exact": exact,
            "python": {
                "value": py_val,
                "abs_error": abs(py_val - exact),
            },
            "np32": {
                "value": np32_val,
                "abs_error": abs(np32_val - exact),
            },
            "np64": {
                "value": np64_val,
                "abs_error": abs(np64_val - exact),
            },
        }

    @staticmethod
    def _exp_sqrt2():
        """sqrt(2)**2 - 2 -> debería ser 0, pero no lo es por redondeo."""
        exact = 0.0

        py_val = (2 ** 0.5) ** 2 - 2
        np32_val = float((np.float32(np.sqrt(2))) ** 2 - np.float32(2))
        np64_val = float((np.float64(np.sqrt(2))) ** 2 - np.float64(2))

        return {
            "title": "sqrt(2)**2 - 2",
            "description": (
                "Ejemplo de error de redondeo: al elevar y restar, el resultado "
                "no es exactamente 0 por la representación binaria de √2."
            ),
            "exact": exact,
            "python": {
                "value": py_val,
                "abs_error": abs(py_val - exact),
            },
            "np32": {
                "value": np32_val,
                "abs_error": abs(np32_val - exact),
            },
            "np64": {
                "value": np64_val,
                "abs_error": abs(np64_val - exact),
            },
        }

    @staticmethod
    def run(n_reps=10):
        """
        Ejecuta todos los experimentos del taller.
        n_reps: número de repeticiones para la suma de 0.1
        """
        experiments = []
        experiments.append(NumpyWorkshop._exp_sum_0_1(n_reps))
        experiments.append(NumpyWorkshop._exp_cancellation())
        experiments.append(NumpyWorkshop._exp_sqrt2())
        return experiments
