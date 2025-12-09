from fractions import Fraction
from typing import List, Tuple, Optional, Dict
import copy

def fraction_to_string(matrix: List[List[Fraction]]) -> List[List[str]]:
    """Convierte una matriz de Fraction a strings para mostrar en HTML."""
    return [[str(val) for val in row] for row in matrix]

def string_to_fraction(matrix_str: List[List[str]]) -> List[List[Fraction]]:
    """Convierte una matriz de strings a Fraction."""
    return [[Fraction(val) for val in row] for row in matrix_str]

def is_square(matrix: List[List[Fraction]]) -> bool:
    return all(len(row) == len(matrix) for row in matrix)

def has_zero_row_or_col(matrix: List[List[Fraction]]) -> bool:
    n = len(matrix)
    for row in matrix:
        if all(x == 0 for x in row):
            return True
    for j in range(n):
        if all(matrix[i][j] == 0 for i in range(n)):
            return True
    return False

def has_proportional_rows(matrix: List[List[Fraction]]) -> bool:
    n = len(matrix)
    for i in range(n):
        for k in range(i + 1, n):
            if matrix[i] and matrix[k]:
                # busca el primer par no nulo para evitar divisiones por cero
                pivot = None
                for j in range(len(matrix[i])):
                    if matrix[k][j] != 0 or matrix[i][j] != 0:
                        pivot = j
                        break
                if pivot is None:
                    # filas completamente nulas (ya capturadas en has_zero_row_or_col)
                    continue
                if matrix[k][pivot] == 0:
                    continue
                ratio = matrix[i][pivot] / matrix[k][pivot]
                if all(matrix[k][j] * ratio == matrix[i][j] for j in range(n)):
                    return True
    return False

def sarrus(matrix: List[List[Fraction]]) -> Tuple[Fraction, List[str]]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("La regla de Sarrus solo aplica a matrices 3x3.")
    
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    steps = [
        f"Matriz A = [[{a}, {b}, {c}], [{d}, {e}, {f}], [{g}, {h}, {i}]]",
        "",
        "Regla de Sarrus: det(A) = (a·e·i + b·f·g + c·d·h) − (c·e·g + b·d·i + a·f·h)",
        "",
        f"Productos principales: {a}·{e}·{i} + {b}·{f}·{g} + {c}·{d}·{h}",
        f"                     = {a*e*i} + {b*f*g} + {c*d*h} = {a*e*i + b*f*g + c*d*h}",
        "",
        f"Productos secundarios: {c}·{e}·{g} + {b}·{d}·{i} + {a}·{f}·{h}",
        f"                     = {c*e*g} + {b*d*i} + {a*f*h} = {c*e*g + b*d*i + a*f*h}",
        "",
        f"det(A) = ({a*e*i + b*f*g + c*d*h}) − ({c*e*g + b*d*i + a*f*h}) = {a*e*i + b*f*g + c*d*h - (c*e*g + b*d*i + a*f*h)}"
    ]
    
    det = (a*e*i + b*f*g + c*d*h) - (c*e*g + b*d*i + a*f*h)
    return det, steps

def minor(matrix: List[List[Fraction]], i: int, j: int) -> List[List[Fraction]]:
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def cofactor(matrix: List[List[Fraction]], i: int, j: int) -> Fraction:
    submatrix = minor(matrix, i, j)
    det_sub = determinant_cofactor(submatrix)[0]  
    return (-1)**(i+j) * det_sub

def determinant_cofactor(matrix: List[List[Fraction]]) -> Tuple[Fraction, List[str]]:
    n = len(matrix)
    if n == 1:
        return matrix[0][0], [f"det({matrix[0][0]}) = {matrix[0][0]}"]
    if n == 2:
        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        steps = [
            f"det([[a, b], [c, d]]) = a·d − b·c",
            f"                     = {matrix[0][0]}·{matrix[1][1]} − {matrix[0][1]}·{matrix[1][0]} = {det}"
        ]
        return det, steps
    
    steps = [f"Expansión por la fila 0:"]
    det = Fraction(0)
    for j in range(n):
        c = cofactor(matrix, 0, j)
        term = matrix[0][j] * c
        det += term
        sign = "+" if (0 + j) % 2 == 0 else "−"
        steps.append(f"  {sign} {matrix[0][j]} · C₀ⱼ = {matrix[0][j]} · ({(-1)**(0+j)}) · det(M₀ⱼ) = {term}")
    
    steps.append(f"det(A) = {det}")
    return det, steps

def cramer_determinant(matrix: List[List[Fraction]]) -> Tuple[Fraction, List[str]]:
    n = len(matrix)
    if n != 3:
        raise ValueError("Método de Cramer ilustrativo solo para 3x3.")
    
    steps = ["Método de Cramer: resolver Ax = 0 con x = [x₁, x₂, x₃]"]
    steps.append("Si det(A) ≠ 0 → solo solución x = 0")
    steps.append("Si det(A) = 0 → infinitas soluciones o ninguna")
    
    det, _ = determinant_cofactor(matrix)  
    steps.append(f"det(A) = {det} (calculado por cofactores)")
    
    if det == 0:
        steps.append("→ Sistema tiene infinitas soluciones → det(A) = 0")
    else:
        steps.append("→ Solo solución trivial → det(A) ≠ 0")
    
    return det, steps

def calculate_determinant(matrix_str: List[List[str]], method: str) -> dict:
    matrix = string_to_fraction(matrix_str)
    
    if not is_square(matrix):
        raise ValueError("La matriz debe ser cuadrada.")
    
    n = len(matrix)
    if n == 0:
        raise ValueError("Matriz vacía.")
    
    det = Fraction(0)
    steps = []
    method_name = ""
    
    if method == "sarrus" and n == 3:
        det, steps = sarrus(matrix)
        method_name = "Regla de Sarrus"
    elif method == "cofactor":
        det, steps = determinant_cofactor(matrix)
        method_name = "Expansión por Cofactores"
    elif method == "cramer" and n == 3:
        det, steps = cramer_determinant(matrix)
        method_name = "Método de Cramer (ilustrativo)"
    else:
        raise ValueError(f"Método no aplicable para matriz {n}x{n}")
    
    invertibility = "El determinante es distinto de cero, por lo tanto A es invertible." if det != 0 else "El determinante es cero, la matriz no tiene inversa (singular)."
    
    properties = []
    
    if has_zero_row_or_col(matrix):
        properties.append("Propiedad 1: Existe una fila o columna nula → det(A) = 0")
    
    if has_proportional_rows(matrix):
        properties.append("Propiedad 2: Dos filas son proporcionales → det(A) = 0")
    
    if n >= 2:
        swapped = copy.deepcopy(matrix)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        det_swapped, _ = determinant_cofactor(swapped)
        if det_swapped == -det:
            properties.append("Propiedad 3: Intercambiar dos filas cambia el signo del determinante")
    
    if n >= 1 and matrix[0][0] != 0:
        k = Fraction(2)
        scaled = copy.deepcopy(matrix)
        scaled[0] = [k * x for x in scaled[0]]
        det_scaled, _ = determinant_cofactor(scaled)
        if det_scaled == k * det:
            properties.append(f"Propiedad 4: Multiplicar fila por {k} → det se multiplica por {k}")
    
    return {
        "det": str(det),
        "steps": steps,
        "method": method_name,
        "invertibility": invertibility,
        "properties": properties,
        "matrix_str": fraction_to_string(matrix)
    }

def _fmt_frac(x: Fraction) -> str:
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{x.numerator}/{x.denominator}"
    return str(x)

def _fmt_matrix_html(mat):
    # arma una tablita simple en HTML
    html = ['<table style="border-collapse:collapse;">']
    for row in mat:
        html.append('<tr>')
        for val in row:
            html.append(
                f'<td style="border:1px solid #ccc;padding:3px 6px;text-align:center;">{_fmt_frac(val)}</td>'
            )
        html.append('</tr>')
    html.append('</table>')
    return "".join(html)

def multiply_matrices_with_steps(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("No se pueden multiplicar: las columnas de A deben ser iguales a las filas de B.")

    steps = []
    steps.append(f"Multiplicación de matrices: A ({rows_A}x{cols_A}) · B ({rows_B}x{cols_B})")
    steps.append("Cada elemento C[i][j] de AB se obtiene como la suma A[i][k]·B[k][j].")

    AB = [[Fraction(0) for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            total = Fraction(0)
            partials = []
            for k in range(cols_A):
                prod = A[i][k] * B[k][j]
                partials.append(
                    f"A[{i}][{k}]·B[{k}][{j}] = {_fmt_frac(A[i][k])}·{_fmt_frac(B[k][j])} = {_fmt_frac(prod)}"
                )
                total += prod
            AB[i][j] = total
            steps.append(
                f"Elemento C[{i}][{j}]: "
                + " + ".join(p.split(" = ")[-1] for p in partials)
                + f" = {_fmt_frac(total)}"
            )
            for p in partials:
                steps.append("   " + p)

    # 👇 aquí metemos la tabla HTML
    steps.append(f"Matriz AB resultante ({rows_A}x{cols_B}):</br>{_fmt_matrix_html(AB)}")

    return AB, steps

def determinant_product_property(matrixA_str: List[List[str]],
                                 matrixB_str: List[List[str]],
                                 method: str = "cofactor") -> dict:
    # Reutilizamos tus helpers
    A = string_to_fraction(matrixA_str)
    B = string_to_fraction(matrixB_str)

    # Validaciones “de libro” para det(AB) = det(A)·det(B)
    # Lo usual: A y B deben ser cuadradas del mismo orden
    if not is_square(A) or not is_square(B):
        raise ValueError("Para verificar det(AB) = det(A)·det(B), A y B deben ser matrices cuadradas.")
    if len(A) != len(B):
        raise ValueError("A y B deben ser del mismo orden para verificar la propiedad 5.")

    n = len(A)

    # 1) LADO IZQUIERDO: det(AB)
    AB, mult_steps = multiply_matrices_with_steps(A, B)

    # elegimos el método; para generalidad usamos cofactores
    if method == "sarrus" and n == 3:
        det_AB, det_AB_steps = sarrus(AB)
        lhs_method_name = "Regla de Sarrus sobre AB"
    else:
        det_AB, det_AB_steps = determinant_cofactor(AB)
        lhs_method_name = "Expansión por cofactores sobre AB"

    lhs_steps = []
    lhs_steps.append("=== LADO IZQUIERDO: det(AB) ===")
    lhs_steps.extend(mult_steps)
    lhs_steps.append(f"Ahora calculamos el determinante de AB usando: {lhs_method_name}")
    lhs_steps.extend(det_AB_steps)
    lhs_steps.append(f"Resultado lado izquierdo: det(AB) = {det_AB}")

    # 2) LADO DERECHO: det(A) · det(B)
    # determinante de A
    if method == "sarrus" and n == 3:
        det_A, det_A_steps = sarrus(A)
        method_A_name = "Regla de Sarrus sobre A"
    else:
        det_A, det_A_steps = determinant_cofactor(A)
        method_A_name = "Expansión por cofactores sobre A"

    # determinante de B
    if method == "sarrus" and n == 3:
        det_B, det_B_steps = sarrus(B)
        method_B_name = "Regla de Sarrus sobre B"
    else:
        det_B, det_B_steps = determinant_cofactor(B)
        method_B_name = "Expansión por cofactores sobre B"

    rhs_steps = []
    rhs_steps.append("=== LADO DERECHO: det(A) × det(B) ===")
    rhs_steps.append(f"Cálculo de det(A) usando: {method_A_name}")
    rhs_steps.extend(det_A_steps)
    rhs_steps.append(f"Cálculo de det(B) usando: {method_B_name}")
    rhs_steps.extend(det_B_steps)
    rhs_steps.append(f"Multiplicamos: det(A) · det(B) = {det_A} · {det_B} = {det_A * det_B}")
    rhs_value = det_A * det_B

    # 3) Armamos el resultado completo
    return {
        "lhs": {
            "matrix": fraction_to_string(AB),
            "det": str(det_AB),
            "steps": lhs_steps
        },
        "rhs": {
            "detA": str(det_A),
            "detB": str(det_B),
            "product": str(rhs_value),
            "steps": rhs_steps
        },
        "same": det_AB == rhs_value,
        "note": "Propiedad 5 verificada" if det_AB == rhs_value else "La igualdad no se cumple (revisa las matrices o el método)."
    }

