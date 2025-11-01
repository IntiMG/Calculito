from fractions import Fraction
from typing import List, Tuple, Optional
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
                ratio = matrix[i][0] / matrix[k][0]
                if all(matrix[i][j] == ratio * matrix[k][j] for j in range(n)):
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