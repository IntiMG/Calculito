# models/numerical_errors.py
"""
Módulo: numerical_errors
------------------------

Aquí se documentan y ejemplifican varios tipos de error numérico
que aparecen cuando la computadora trabaja con números reales.

TIPOS DE ERROR EXPLICADOS:

1. Error inherente:
   - Es el error que ya viene "de fábrica" en los datos de entrada.
   - Por ejemplo, si la tasa de interés real es 6.247% pero el banco
     la publica como 6.25%, ya hay una diferencia antes de calcular nada.
   - En código no lo "creamos", solo lo reconocemos: los datos que nos
     da el usuario casi nunca son perfectos.

2. Error de redondeo:
   - Se produce cuando un número se ajusta a un número cercano
     porque el formato no puede representar todos los decimales.
   - Ejemplo: 9.9609375 se redondea a 9.96 (2 decimales).
   - En la computadora, los números flotantes se redondean de forma
     automática porque solo hay un número finito de bits.

3. Error de truncamiento:
   - Se produce cuando cortamos los decimales "sin redondear".
   - Ejemplo: truncar 9.9609375 a 2 decimales da 9.96,
     truncar 3.999 a 1 decimal da 3.9.
   - Es típico cuando decidimos trabajar solo con cierto número de
     cifras significativas o decimales.

4. Overflow y underflow:
   - Overflow: el número es tan grande que ya no cabe en el tipo
     de dato flotante → Python lo representa como "inf".
   - Underflow: el número es tan pequeño (cerca de 0) que se
     aproxima a 0 porque ya no se pueden representar más dígitos.
   - Estos errores se dan en procesos iterativos con potencias muy
     grandes o muy pequeñas.

5. Error del modelo matemático:
   - No depende de la computadora, sino de cómo modelamos la realidad.
   - Ejemplo: usar interés compuesto mensual para algo que en la vida
     real no crece exactamente de forma exponencial.
   - Aunque los cálculos estén "bien", el modelo puede no representar
     bien la situación real.
"""

from dataclasses import dataclass
from typing import List, Literal
import math


def truncate(value: float, decimals: int) -> float:
    """
    Trunca un número a 'decimals' decimales SIN redondear.

    Ejemplo:
        truncate(9.9609375, 2) -> 9.96
    """
    factor = 10 ** decimals
    return math.trunc(value * factor) / factor


@dataclass
class IterationRow:
    """
    Representa una fila de la tabla de acumulación de error.

    Campos:
        iteracion       → número de iteración (1, 2, 3, ...)
        monto_anterior  → capital con el que se inicia la iteración
        interes_aprox   → interés calculado con truncamiento o redondeo
        interes_real    → interés "ideal" con todos los decimales
        monto_aprox     → nuevo capital aproximado (monto_anterior + interes_aprox)
        monto_real      → nuevo capital real   (monto_real_anterior + interes_real)
        diferencia      → |monto_real - monto_aprox| (error absoluto de esa iteración)
        error_acumulado → suma de las diferencias hasta esta iteración
    """
    iteracion: int
    monto_anterior: float
    interes_aprox: float
    interes_real: float
    monto_aprox: float
    monto_real: float
    diferencia: float
    error_acumulado: float


class ErrorAnalysis:
    """
    Clase con utilidades para analizar errores numéricos.

    El método principal para el taller guiado es:

        generar_tabla_interes_compuesto(...)

    que construye una tabla similar a la de la guía del profesor,
    mostrando cómo se acumula el error de truncamiento o de redondeo.
    """

    @staticmethod
    def generar_tabla_interes_compuesto(
        capital_inicial: float,
        tasa: float,
        iteraciones: int,
        decimales: int = 2,
        tipo_error: Literal["truncamiento", "redondeo"] = "truncamiento",
    ) -> List[IterationRow]:
        """
        Genera una tabla de interés compuesto comparando el valor real
        vs. el valor aproximado (por truncamiento o redondeo).

        Parámetros:
            capital_inicial → capital de la primera iteración (ej. 150000)
            tasa            → tasa en forma decimal (ej. 0.0625 para 6.25%)
            iteraciones     → número de periodos a simular (mínimo 1)
            decimales       → decimales que se mantienen en el interés aproximado
            tipo_error      → 'truncamiento' o 'redondeo'

        Retorna:
            Lista de IterationRow para que Flask la muestre en una tabla.
        """

        if iteraciones <= 0:
            raise ValueError("El número de iteraciones debe ser mayor que cero.")

        if tipo_error not in ("truncamiento", "redondeo"):
            raise ValueError("Tipo de error no válido. Use 'truncamiento' o 'redondeo'.")

        filas: List[IterationRow] = []

        # capital "real" y "aproximado" al inicio
        monto_real = float(capital_inicial)
        monto_aprox = float(capital_inicial)

        error_acumulado = 0.0

        for k in range(1, iteraciones + 1):
            # 1) Interés real (sin recortar decimales)
            interes_real = monto_real * tasa

            # 2) Interés aproximado (truncado o redondeado)
            interes_bruto = monto_aprox * tasa
            if tipo_error == "truncamiento":
                interes_aprox = truncate(interes_bruto, decimales)
            else:
                interes_aprox = round(interes_bruto, decimales)

            # 3) Nuevos montos
            nuevo_monto_real = monto_real + interes_real
            nuevo_monto_aprox = monto_aprox + interes_aprox

            # 4) Error de esta iteración y error acumulado
            diferencia = abs(nuevo_monto_real - nuevo_monto_aprox)
            error_acumulado += diferencia

            filas.append(
                IterationRow(
                    iteracion=k,
                    monto_anterior=monto_aprox,   # lo que el usuario "ve" como capital anterior
                    interes_aprox=interes_aprox,
                    interes_real=interes_real,
                    monto_aprox=nuevo_monto_aprox,
                    monto_real=nuevo_monto_real,
                    diferencia=diferencia,
                    error_acumulado=error_acumulado,
                )
            )

            # Actualizar para la siguiente iteración
            monto_real = nuevo_monto_real
            monto_aprox = nuevo_monto_aprox

        return filas
