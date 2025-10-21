from flask import Flask, render_template, request, url_for, session
from models.equations_solver import Gauss
from models.properties import Properties
from models.matrix_equation import MatrixEquation
from models.arithmetic_operations import ArOperations
from fractions import Fraction

app = Flask(__name__)
app.secret_key = 'a_secret_key'  # Required for session

# ========= Filtros Jinja para formatear fracciones y vectores =========
def _fmt_num(x, tol=1e-9):
    # Fraction → "a/b" o "a" si es entero
    if hasattr(x, "numerator") and hasattr(x, "denominator"):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{x.numerator}/{x.denominator}"
    # int/float → entero si está "casi" entero
    try:
        xf = float(x)
        if abs(xf - round(xf)) < tol:
            return str(int(round(xf)))
        # evitar "-0"
        if abs(xf) < tol:
            return "0"
        return f"{xf:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def safe_fraction(val):
    if val is None:
        return Fraction(0)
    s = str(val).strip()
    if s == "":
        return Fraction(0)
    return Fraction(s)

# ========= Helpers para Gauss-Jordan (sin NumPy) =========
from copy import deepcopy

def identity(n):
    from fractions import Fraction
    I = [[Fraction(0) for _ in range(n)] for __ in range(n)]
    for i in range(n):
        I[i][i] = Fraction(1)
    return I

def augmented(A, B):
    return [rowA + rowB for rowA, rowB in zip(A, B)]

def split_augmented(M, ncols_left):
    left = [row[:ncols_left] for row in M]
    right = [row[ncols_left:] for row in M]
    return left, right

def is_identity(M):
    from fractions import Fraction
    n = len(M)
    for i in range(n):
        for j in range(n):
            if M[i][j] != (Fraction(1) if i == j else Fraction(0)):
                return False
    return True

def gauss_jordan_with_steps(A):
    """
    Aplica Gauss-Jordan a [A | I]. Devuelve:
      ok, steps, left, right, pivots, rank, reason
    steps: lista de dicts con 'description', 'matrix' (lado A), 'results' (lado I/A⁻¹) para que tu UI lo muestre como [A|B]
    """
    from fractions import Fraction
    n = len(A)
    I = identity(n)
    aug = augmented(deepcopy(A), I)
    steps = []

    def fmt_num_local(x):
        # usa tu mismo criterio visual
        return jinja_fmt_num(x) if callable(jinja_fmt_num) else str(x)

    def snapshot(desc):
        left, right = split_augmented(aug, n)
        steps.append({
            "description": desc,
            "matrix": [[fmt_num_local(v) for v in row] for row in left],
            # tu viewer espera “results” por fila; convertimos cada fila de right a string compacto
            "results": [" ".join(fmt_num_local(x) for x in row) for row in right]
        })

    snapshot("Construcción de la matriz aumentada [A | I].")

    row = 0
    pivots = []
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            aug[row], aug[pivot] = aug[pivot], aug[row]
            snapshot(f"Intercambiar R{row+1} ↔ R{pivot+1}")

        pv = aug[row][col]
        if pv != 1:
            for j in range(2*n):
                aug[row][j] /= pv
            snapshot(f"Escalar R{row+1} ← R{row+1} / {fmt_num_local(pv)}")

        for r in range(n):
            if r != row and aug[r][col] != 0:
                factor = aug[r][col]
                for j in range(2*n):
                    aug[r][j] -= factor * aug[row][j]
                snapshot(f"R{r+1} ← R{r+1} - ({fmt_num_local(factor)})·R{row+1}")

        pivots.append((row, col))
        row += 1
        if row == n:
            break

    rank = len(pivots)
    left, right = split_augmented(aug, n)

    if not is_identity(left):
        reason = "La matriz no es invertible porque no tiene pivote en cada fila."
        snapshot("No se logró obtener I en el lado izquierdo; A no es invertible.")
        return False, steps, left, right, pivots, rank, reason

    snapshot("Se obtuvo [I | A⁻¹]. La matriz derecha es A⁻¹.")
    return True, steps, left, right, pivots, rank, ""

def props_invertibilidad(n, rank):
    """
    Verifica (c)(d)(e) y devuelve banderas + interpretación corta.
    """
    interp = {
        "c": "Si A tiene n pivotes, entonces A es invertible.",
        "d": "Si A x = 0 solo tiene la solución trivial, entonces A⁻¹ existe.",
        "e": "Si las columnas son linealmente independientes, entonces A es invertible."
    }
    ok = (rank == n)
    return {
        "c": {"ok": ok, "text": interp["c"]},
        "d": {"ok": ok, "text": interp["d"]},
        "e": {"ok": ok, "text": interp["e"]},
    }


@app.template_filter("fmt_num")
def jinja_fmt_num(x):
    return _fmt_num(x)

@app.template_filter("fmt_vec")
def jinja_fmt_vec(vec):
    return "[" + ", ".join(_fmt_num(v) for v in vec) + "]"


# ========= Rutas Existentes =========
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# Ruta para sistemas de ecuaciones lineales
@app.route("/linear_system", methods=["GET", "POST"])
def linear_system():
    if request.method == "POST":
        num_vars = request.form.get("num_vars")
        num_eqs = request.form.get("num_eqs")

        if num_vars and num_eqs:
            try:
                num_vars = int(num_vars)
                num_eqs = int(num_eqs)
                if num_vars <= 0 or num_eqs <= 0 or num_vars > 10 or num_eqs > 10:
                    return render_template("index.html", step=1, error="El número de variables y ecuaciones debe ser entre 1 y 10.")
                return render_template("index.html", num_vars=num_vars, num_eqs=num_eqs, step=2)
            except ValueError:
                return render_template("index.html", step=1, error="Por favor, ingrese números válidos.")
        else:
            return render_template("index.html", step=1, error="Por favor, complete todos los campos.")

    return render_template("index.html", step=1)

@app.route("/solve", methods=["POST"])
def solve_linear_system():
    num_vars = request.form.get("num_vars")
    num_eqs = request.form.get("num_eqs")

    if not num_vars or not num_eqs:
        return render_template("index.html", step=1, error="Error: Número de variables o ecuaciones no proporcionado.")

    try:
        num_vars = int(num_vars)
        num_eqs = int(num_eqs)
        if num_vars <= 0 or num_eqs <= 0:
            return render_template("index.html", step=1, error="El número de variables y ecuaciones debe ser mayor que 0.")
    except ValueError:
        return render_template("index.html", step=1, error="Error: Número de variables o ecuaciones no válido.")

    matrix = []
    try:
        for i in range(num_eqs):
            row = []
            for j in range(num_vars + 1):
                val = request.form.get(f"cell_{i}_{j}")
                if val is None:
                    return render_template("index.html", step=1, error=f"Error: Falta el valor en la celda ({i}, {j}).")
                row.append(float(val))
            matrix.append(row)
    except ValueError:
        return render_template("index.html", step=1, error="Error: Ingrese valores numéricos válidos en la matriz.")

    coefficients = [row[:-1] for row in matrix]
    results = [row[-1] for row in matrix]

    if len(coefficients) != num_eqs or any(len(row) != num_vars for row in coefficients):
        return render_template("index.html", step=1, error="Error: Dimensiones de la matriz no válidas.")
    if len(results) != num_eqs:
        return render_template("index.html", step=1, error="Error: Vector de resultados no válido.")

    try:
        gauss_solver = Gauss(coefficients, results, use_fractions=True)
        solution = gauss_solver.get_formatted_solution()
        steps = gauss_solver.get_steps()
        info = gauss_solver.get_classification()
        pivot_report = gauss_solver.get_pivot_report()

        return render_template(
            "result.html",
            solution=solution,
            steps=steps,
            consistent=info["consistent"],
            tipo=("Única" if info["status"] == "unique" else ("Infinitas" if info["status"] == "infinite" else "Ninguna")),
            rank=info["rank"],
            n=info["n"],
            pivot_report=pivot_report,
            back_url=url_for("linear_system")
        )
    except Exception as e:
        error_msg = str(e) if "No tiene solución" in str(e) else "No tiene solución"
        return render_template("index.html", step=1, error=f"Error al resolver el sistema: {error_msg}")

# Ruta para propiedades algebraicas
@app.route("/properties", methods=["GET", "POST"])
def properties():
    if request.method == "POST":
        dimension = request.form.get("dimension")

        if dimension:
            try:
                dimension = int(dimension)
                if dimension <= 0 or dimension > 10:
                    return render_template("properties.html", step=1, error="La dimensión debe ser entre 1 y 10.")
                return render_template("properties.html", dimension=dimension, step=2)
            except ValueError:
                return render_template("properties.html", step=1, error="Por favor, ingrese un número válido.")
        else:
            return render_template("properties.html", step=1, error="Por favor, complete el campo.")

    return render_template("properties.html", step=1)

@app.route("/compute_properties", methods=["POST"])
def compute_properties():
    dimension = int(request.form["dimension"])
    try:
        u = [float(request.form[f"u_{i}"]) for i in range(dimension)]
        v = [float(request.form[f"v_{i}"]) for i in range(dimension)]
        scalar = float(request.form["scalar"])
    except (ValueError, KeyError):
        return render_template("properties.html", step=1, error="Error: Ingrese valores numéricos válidos.")

    try:
        props = Properties(u, v, scalar, dimension, use_fractions=True)
        verifications = props.get_verifications()
        computations = props.get_computations()

        return render_template("properties_result.html", verifications=verifications, computations=computations)
    except Exception as e:
        return render_template("properties.html", step=1, error=f"Error: {str(e)}")

# Ruta para combinación lineal
@app.route("/linear_combination", methods=["GET", "POST"])
def linear_combination():
    if request.method == "POST":
        dimension = request.form.get("dimension")
        num_vectors = request.form.get("num_vectors")

        if dimension and num_vectors:
            try:
                dimension = int(dimension)
                num_vectors = int(num_vectors)
                if dimension <= 0 or num_vectors <= 0 or dimension > 10 or num_vectors > 10:
                    return render_template("linear_combination.html", step=1, error="Los valores deben ser entre 1 y 10.")
                return render_template("linear_combination.html", dimension=dimension, num_vectors=num_vectors, step=2)
            except ValueError:
                return render_template("linear_combination.html", step=1, error="Por favor, ingrese números válidos.")
        else:
            return render_template("linear_combination.html", step=1, error="Por favor, complete todos los campos.")

    return render_template("linear_combination.html", step=1)

@app.route("/solve_linear_combination", methods=["POST"])
def solve_linear_combination():
    dimension = int(request.form["dimension"])
    num_vectors = int(request.form["num_vectors"])

    try:
        coefficients = [[float(request.form[f"v_{i}_{j}"]) for j in range(num_vectors)] for i in range(dimension)]
        results = [float(request.form[f"b_{i}"]) for i in range(dimension)]
    except (ValueError, KeyError):
        return render_template("linear_combination.html", step=1, error="Error: Ingrese valores numéricos válidos.")

    try:
        gauss_solver = Gauss(coefficients, results, use_fractions=True)
        solution = gauss_solver.get_formatted_solution()
        steps = gauss_solver.get_steps()
        info = gauss_solver.get_classification()
        pivot_report = gauss_solver.get_pivot_report()

        is_combination = info["consistent"]
        interpretation = "El vector objetivo es una combinación lineal." if is_combination else "El vector objetivo NO es una combinación lineal."

        return render_template(
            "result.html",
            solution=solution,
            steps=steps,
            consistent=info["consistent"],
            tipo=("Única" if info["status"] == "unique" else ("Infinitas" if info["status"] == "infinite" else "Ninguna")),
            rank=info["rank"],
            n=info["n"],
            pivot_report=pivot_report,
            interpretation=interpretation,
            back_url=url_for("linear_combination")
        )
    except Exception as e:
        error_msg = str(e) if "No tiene solución" in str(e) else "No tiene solución"
        return render_template("linear_combination.html", step=1, error=f"Error: {error_msg}")

# Ruta para ecuación vectorial
@app.route("/vector_equation", methods=["GET", "POST"])
def vector_equation():
    if request.method == "POST":
        dimension = request.form.get("dimension")
        num_vectors = request.form.get("num_vectors")

        if dimension and num_vectors:
            try:
                dimension = int(dimension)
                num_vectors = int(num_vectors)
                if dimension <= 0 or num_vectors <= 0 or dimension > 10 or num_vectors > 10:
                    return render_template("vector_equation.html", step=1, error="Los valores deben ser entre 1 y 10.")
                return render_template("vector_equation.html", dimension=dimension, num_vectors=num_vectors, step=2)
            except ValueError:
                return render_template("vector_equation.html", step=1, error="Por favor, ingrese números válidos.")
        else:
            return render_template("vector_equation.html", step=1, error="Por favor, complete todos los campos.")

    return render_template("vector_equation.html", step=1)

@app.route("/solve_vector_equation", methods=["POST"])
def solve_vector_equation():
    dimension = int(request.form["dimension"])
    num_vectors = int(request.form["num_vectors"])

    try:
        coefficients = [[safe_fraction(request.form.get(f"v_{i}_{j}"))
                         for j in range(num_vectors)] for i in range(dimension)]
        results = [safe_fraction(request.form.get(f"b_{i}")) for i in range(dimension)]
    except Exception:
        return render_template("vector_equation.html", step=1, error="Error: Ingrese valores válidos (admite fracciones tipo 3/2).")

    try:
        gauss_solver = Gauss(coefficients, results, use_fractions=True)
        solution_lines = gauss_solver.get_formatted_solution()
        steps = gauss_solver.get_steps()
        info = gauss_solver.get_classification()
        pivot_report = gauss_solver.get_pivot_report()

        tipo = "Única" if info["status"] == "unique" else ("Infinitas" if info["status"] == "infinite" else "Ninguna")
        consistent = info["status"] != "inconsistent"
        interpretation = (
            "Existe una única combinación de los vectores que produce b." if info["status"] == "unique" else
            "Existen infinitas combinaciones (parámetros libres)." if info["status"] == "infinite" else
            "No hay combinación de los vectores que produzca b."
        )

        comb_expr = None
        if info["status"] == "unique":
            coeffs = gauss_solver.solution
            def fmt(fr):
                if hasattr(fr, "denominator") and fr.denominator == 1:
                    return str(fr.numerator)
                if hasattr(fr, "numerator"):
                    return f"{fr.numerator}/{fr.denominator}"
                return f"{float(fr):.6f}".rstrip("0").rstrip(".")
            terms = [f"({fmt(coeffs[j])})·v{j+1}" for j in range(info["n"])]
            comb_expr = "b = " + " + ".join(terms)

        return render_template(
            "result.html",
            solution=solution_lines,
            steps=steps,
            consistent=consistent,
            tipo=tipo,
            rank=info["rank"],
            n=info["n"],
            pivot_report=pivot_report,
            interpretation=interpretation,
            comb_expr=comb_expr,
            back_url=url_for("vector_equation")
        )
    except Exception as e:
        return render_template("vector_equation.html", step=1, error=f"Error: {str(e)}")

# Ruta para ecuación matricial
@app.route("/matrix_equation", methods=["GET", "POST"])
def matrix_equation():
    if request.method == "POST":
        rows_a = request.form.get("rows_a")
        cols_a = request.form.get("cols_a")
        cols_b = request.form.get("cols_b")

        if rows_a and cols_a and cols_b:
            try:
                rows_a = int(rows_a)
                cols_a = int(cols_a)
                cols_b = int(cols_b)
                if rows_a <= 0 or cols_a <= 0 or cols_b <= 0 or rows_a > 10 or cols_a > 10 or cols_b > 10:
                    return render_template("matrix_form.html", step=1, error="Los valores deben ser entre 1 y 10.")
                return render_template("matrix_form.html", rows_a=rows_a, cols_a=cols_a, cols_b=cols_b, step=2)
            except ValueError:
                return render_template("matrix_form.html", step=1, error="Por favor, ingrese números válidos.")
        else:
            return render_template("matrix_form.html", step=1, error="Por favor, complete todos los campos.")

    return render_template("matrix_form.html", step=1)

@app.route("/solve_matrix_equation", methods=["POST"])
def solve_matrix_equation():
    rows_a = int(request.form["rows_a"])
    cols_a = int(request.form["cols_a"])
    cols_b = int(request.form["cols_b"])

    try:
        A = [[float(request.form[f"a_{i}_{j}"]) for j in range(cols_a)] for i in range(rows_a)]
        B = [[float(request.form[f"b_{i}_{k}"]) for k in range(cols_b)] for i in range(rows_a)]
    except (ValueError, KeyError):
        return render_template("matrix_form.html", step=1, error="Error: Ingrese valores numéricos válidos.")

    try:
        matrix_solver = MatrixEquation(A, B, use_fractions=True)
        solutions = matrix_solver.get_formatted_solutions()
        steps = matrix_solver.get_all_steps()
        infos = matrix_solver.infos
        pivot_reports = matrix_solver.get_all_pivot_reports()
        overall_info = matrix_solver.get_overall_classification()

        return render_template(
            "matrix_result.html",
            solutions=solutions,
            steps=steps,
            infos=infos,
            pivot_reports=pivot_reports,
            overall_info=overall_info
        )
    except Exception as e:
        error_msg = str(e)
        return render_template("matrix_form.html", step=1, error=f"Error: {error_msg}")

# Ruta para sistema homogéneo y dependencia lineal
@app.route("/homogeneous", methods=["GET", "POST"])
def homogeneous():
    if request.method == "POST":
        rows = request.form.get("rows")
        cols = request.form.get("cols")
        if rows and cols:
            try:
                rows = int(rows)
                cols = int(cols)
                if not (1 <= rows <= 10 and 1 <= cols <= 10):
                    raise ValueError()
                return render_template("homogeneous.html", step=2, rows=rows, cols=cols)
            except Exception:
                return render_template("homogeneous.html", step=1, error="Ingresa tamaños válidos (1–10).")
        return render_template("homogeneous.html", step=1, error="Completa ambos campos.")
    return render_template("homogeneous.html", step=1)

@app.route("/solve_homogeneous", methods=["POST"])
def solve_homogeneous():
    rows = int(request.form["rows"])
    cols = int(request.form["cols"])
    try:
        A = [[safe_fraction(request.form.get(f"a_{i}_{j}")) for j in range(cols)] for i in range(rows)]
        b = [safe_fraction(request.form.get(f"b_{i}")) for i in range(rows)]

        is_homogeneous = all(bi == 0 for bi in b)

        gauss = Gauss(A, b, use_fractions=True)
        solution_lines = gauss.get_formatted_solution()
        steps = gauss.get_steps()
        info = gauss.get_classification()
        pivot_report = gauss.get_pivot_report()

        if is_homogeneous:
            if info["status"] == "unique":
                interpretation = "Sistema homogéneo (b = 0) con única solución: la trivial (x = 0)."
            elif info["status"] == "infinite":
                interpretation = "Sistema homogéneo (b = 0) con soluciones no triviales (infinitas)."
            else:
                interpretation = "Sistema homogéneo (b = 0) inconsistente (caso patológico)."
        else:
            if info["status"] == "unique":
                interpretation = "Sistema NO homogéneo (b ≠ 0) con solución única."
            elif info["status"] == "infinite":
                interpretation = "Sistema NO homogéneo (b ≠ 0) con infinitas soluciones (familia afín)."
            else:
                interpretation = "Sistema NO homogéneo (b ≠ 0) inconsistente (no tiene solución)."

        dependence = (
            "Los vectores son linealmente DEPENDIENTES (existen soluciones no triviales en el sistema homogéneo)."
            if info["rank"] < cols
            else "Los vectores son linealmente INDEPENDIENTES (solución única o ninguna en el sistema homogéneo)."
        )

        return render_template(
            "homogeneous.html",
            step=3,
            rows=rows,
            cols=cols,
            solution=solution_lines,
            steps=steps,
            consistent=info["consistent"],
            tipo=("Única" if info["status"] == "unique" else ("Infinitas" if info["status"] == "infinite" else "Ninguna")),
            rank=info["rank"],
            n=info["n"],
            pivot_report=pivot_report,
            interpretation=interpretation,
            dependence=dependence
        )
    except Exception as e:
        return render_template("homogeneous.html", step=1, error=f"Revisa entradas: {e}")

# Ruta para operaciones con vectores
@app.route("/vector_ops", methods=["GET", "POST"])
def vector_ops():
    if request.method == "POST":
        stage = request.form.get("stage", "1")

        if stage == "1":
            try:
                dimension = int(request.form["dimension"])
                num_vectors = int(request.form["num_vectors"])
                if not (1 <= dimension <= 10 and 1 <= num_vectors <= 10):
                    raise ValueError()
            except Exception:
                return render_template("vector_ops.html", step=1,
                                      error="Dimensión y # de vectores deben ser enteros entre 1 y 10.")
            return render_template("vector_ops.html", step=2,
                                  dimension=dimension, num_vectors=num_vectors)

        if stage == "2":
            try:
                dimension = int(request.form["dimension"])
                num_vectors = int(request.form["num_vectors"])

                vectors, coefs = [], []
                for j in range(num_vectors):
                    vec = []
                    for i in range(dimension):
                        val = request.form.get(f"v_{i}_{j}", "").strip()
                        vec.append(Fraction(val))
                    vectors.append(vec)
                    a_val = request.form.get(f"a_{j}", "1").strip()
                    coefs.append(Fraction(a_val))

                result, steps = Properties.linear_combo_with_steps(vectors, coefs, use_fractions=True)

                return render_template("vector_ops.html", step=3,
                                      dimension=dimension, num_vectors=num_vectors,
                                      vectors=vectors, coefs=coefs,
                                      result=result, steps=steps)
            except Exception as e:
                return render_template("vector_ops.html", step=1, error=f"Revisa entradas: {e}")

    return render_template("vector_ops.html", step=1)

# Ruta para multiplicación matriz por vector
@app.route("/matvec", methods=["GET", "POST"])
def matvec():
    if request.method == "POST":
        stage = request.form.get("stage", "1")

        if stage == "1":
            try:
                rows = int(request.form["rows"])
                cols = int(request.form["cols"])
                if not (1 <= rows <= 10 and 1 <= cols <= 10):
                    raise ValueError()
            except Exception:
                return render_template("matvec.html", step=1,
                                      error="Filas y columnas deben ser enteros entre 1 y 10.")
            return render_template("matvec.html", step=2, rows=rows, cols=cols)

        if stage == "2":
            try:
                rows = int(request.form["rows"])
                cols = int(request.form["cols"])
                A = [[safe_fraction(request.form.get(f"a_{i}_{j}")) for j in range(cols)]
                     for i in range(rows)]
                v = [safe_fraction(request.form.get(f"v_{j}")) for j in range(cols)]

                res, steps = Properties.mat_vec_with_steps(A, v, use_fractions=True)

                return render_template(
                    "matvec.html",
                    step=3, rows=rows, cols=cols,
                    A=A, vec=v, result=res, steps=steps
                )
            except Exception as e:
                return render_template("matvec.html", step=1, error=f"Revisa entradas: {e}")

    return render_template("matvec.html", step=1)

# ========= Nuevas Rutas para Operaciones con Matrices =========
@app.route("/matrix_operations", methods=["GET"])
def matrix_operations():
    return render_template("matrix_operations.html")

@app.route("/matrix_add", methods=["GET", "POST"])
def matrix_add():
    use_result = request.args.get('use_result')
    if use_result and 'current_matrix' in session:
        matrix_a = session['current_matrix']
        rows = len(matrix_a)
        cols = len(matrix_a[0])
        return render_template("matrix_add.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, using_result=True)
    if request.method == "POST":
        try:
            rows = int(request.form["rows"])
            cols = int(request.form["cols"])
            if not (1 <= rows <= 10 and 1 <= cols <= 10):
                return render_template("matrix_add.html", step=1, error="Las dimensiones deben estar entre 1 y 10.")
            return render_template("matrix_add.html", step=2, rows=rows, cols=cols)
        except ValueError:
            return render_template("matrix_add.html", step=1, error="Por favor, ingrese dimensiones válidas.")
    return render_template("matrix_add.html", step=1)

@app.route("/matrix_add_solve", methods=["POST"])
def matrix_add_solve():
    try:
        rows = int(request.form["rows"])
        cols = int(request.form["cols"])
        matrix_a = [[safe_fraction(request.form.get(f"A_{i}_{j}", 0)) for j in range(cols)] for i in range(rows)]
        matrix_b = [[safe_fraction(request.form.get(f"B_{i}_{j}", 0)) for j in range(cols)] for i in range(rows)]
        result, steps = ArOperations.addTwoMatrixWithSteps(matrix_a, matrix_b)
        # Convert Fraction objects to strings before storing in session
        session['current_matrix'] = fraction_to_string(result)
        session['previous_op'] = 'add'
        session['original_a'] = fraction_to_string(matrix_a)
        session['original_b'] = fraction_to_string(matrix_b)
        return render_template("matrix_add.html", step=3, rows=rows, cols=cols, matrix_a=matrix_a, matrix_b=matrix_b, result=result, steps=steps)
    except ValueError as e:
        return render_template("matrix_add.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, matrix_b=matrix_b, error=f"Error: Ingrese valores válidos (admite fracciones tipo 3/2). {str(e)}")
    except Exception as e:
        return render_template("matrix_add.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, matrix_b=matrix_b, error=f"Error al realizar la suma: {str(e)}")
    
@app.route("/matrix_subtract", methods=["GET", "POST"])
def matrix_subtract():
    use_result = request.args.get('use_result')
    if use_result and 'current_matrix' in session:
        matrix_a = session['current_matrix']
        rows = len(matrix_a)
        cols = len(matrix_a[0])
        return render_template("matrix_subtract.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, using_result=True)
    if request.method == "POST":
        try:
            rows = int(request.form["rows"])
            cols = int(request.form["cols"])
            if not (1 <= rows <= 10 and 1 <= cols <= 10):
                return render_template("matrix_subtract.html", step=1, error="Las dimensiones deben estar entre 1 y 10.")
            return render_template("matrix_subtract.html", step=2, rows=rows, cols=cols)
        except ValueError:
            return render_template("matrix_subtract.html", step=1, error="Por favor, ingrese dimensiones válidas.")
    return render_template("matrix_subtract.html", step=1)

@app.route("/matrix_subtract_solve", methods=["POST"])
def matrix_subtract_solve():
    try:
        rows = int(request.form["rows"])
        cols = int(request.form["cols"])
        matrix_a = [[safe_fraction(request.form.get(f"A_{i}_{j}", 0)) for j in range(cols)] for i in range(rows)]
        matrix_b = [[safe_fraction(request.form.get(f"B_{i}_{j}", 0)) for j in range(cols)] for i in range(rows)]
        result, steps = ArOperations.subtractTwoMatrixWithSteps(matrix_a, matrix_b)
        session['current_matrix'] = fraction_to_string(result)
        session['previous_op'] = 'subtract'
        session['original_a'] = fraction_to_string(matrix_a)
        session['original_b'] = fraction_to_string(matrix_b)
        return render_template("matrix_subtract.html", step=3, rows=rows, cols=cols, matrix_a=matrix_a, matrix_b=matrix_b, result=result, steps=steps)
    except ValueError as e:
        return render_template("matrix_subtract.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, matrix_b=matrix_b, error=f"Error: Ingrese valores válidos (admite fracciones tipo 3/2). {str(e)}")
    except Exception as e:
        return render_template("matrix_subtract.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, matrix_b=matrix_b, error=f"Error al realizar la resta: {str(e)}")
    
@app.route("/matrix_scalar", methods=["GET", "POST"])
def matrix_scalar():
    use_result = request.args.get('use_result')
    if use_result and 'current_matrix' in session:
        matrix_a = session['current_matrix']
        rows = len(matrix_a)
        cols = len(matrix_a[0])
        return render_template("matrix_scalar.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, using_result=True)
    if request.method == "POST":
        try:
            rows = int(request.form["rows"])
            cols = int(request.form["cols"])
            if not (1 <= rows <= 10 and 1 <= cols <= 10):
                return render_template("matrix_scalar.html", step=1, error="Las dimensiones deben estar entre 1 y 10.")
            return render_template("matrix_scalar.html", step=2, rows=rows, cols=cols)
        except ValueError:
            return render_template("matrix_scalar.html", step=1, error="Por favor, ingrese dimensiones válidas.")
    return render_template("matrix_scalar.html", step=1)

@app.route("/matrix_scalar_solve", methods=["POST"])
def matrix_scalar_solve():
    try:
        rows = int(request.form["rows"])
        cols = int(request.form["cols"])
        scalar = safe_fraction(request.form.get("scalar"))
        matrix_a = [[safe_fraction(request.form.get(f"A_{i}_{j}", 0)) for j in range(cols)] for i in range(rows)]
        result, steps = ArOperations.multiplyMatrixByScalarWithSteps(matrix_a, scalar)
        session['current_matrix'] = fraction_to_string(result)
        session['previous_op'] = 'scalar'
        session['original_a'] = fraction_to_string(matrix_a)
        session['original_scalar'] = str(scalar)  # Convert scalar to string
        return render_template("matrix_scalar.html", step=3, rows=rows, cols=cols, scalar=scalar, matrix_a=matrix_a, result=result, steps=steps)
    except ValueError as e:
        return render_template("matrix_scalar.html", step=2, rows=rows, cols=cols, scalar=scalar, matrix_a=matrix_a, error=f"Error: Ingrese valores válidos (admite fracciones tipo 3/2). {str(e)}")
    except Exception as e:
        return render_template("matrix_scalar.html", step=2, rows=rows, cols=cols, scalar=scalar, matrix_a=matrix_a, error=f"Error al realizar la multiplicación por escalar: {str(e)}")
    
@app.route("/matrix_multiply", methods=["GET", "POST"])
def matrix_multiply():
    use_result = request.args.get('use_result')
    if use_result and 'current_matrix' in session:
        matrix_a = session['current_matrix']  # Esto es una lista de strings
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0]) if matrix_a else 0
        # Solicitar al usuario las dimensiones de la segunda matriz
        return render_template("matrix_multiply.html", step=1, rows_a=rows_a, cols_a=cols_a, using_result=True)
    if request.method == "POST":
        try:
            rows_a = int(request.form["rows_a"])
            cols_a = int(request.form["cols_a"])
            rows_b = int(request.form["rows_b"])
            cols_b = int(request.form["cols_b"])
            if not (1 <= rows_a <= 10 and 1 <= cols_a <= 10 and 1 <= rows_b <= 10 and 1 <= cols_b <= 10):
                return render_template("matrix_multiply.html", step=1, error="Las dimensiones deben estar entre 1 y 10.")
            if cols_a != rows_b:
                return render_template("matrix_multiply.html", step=1, error="Las columnas de A deben ser iguales a las filas de B.")
            return render_template("matrix_multiply.html", step=2, rows_a=rows_a, cols_a=cols_a, rows_b=rows_b, cols_b=cols_b)
        except ValueError:
            return render_template("matrix_multiply.html", step=1, error="Por favor, ingrese dimensiones válidas.")
    return render_template("matrix_multiply.html", step=1)

@app.route("/matrix_multiply_solve", methods=["POST"])
def matrix_multiply_solve():
    try:
        rows_a = int(request.form["rows_a"])
        cols_a = int(request.form["cols_a"])
        rows_b = int(request.form["rows_b"])
        cols_b = int(request.form["cols_b"])
        matrix_a = [[safe_fraction(request.form.get(f"A_{i}_{j}", 0)) for j in range(cols_a)] for i in range(rows_a)]
        matrix_b = [[safe_fraction(request.form.get(f"B_{i}_{j}", 0)) for j in range(cols_b)] for i in range(rows_b)]
        result, steps = ArOperations.multiplyTwoMatrixWithSteps(matrix_a, matrix_b)
        # Convertir el resultado y las matrices originales a cadenas antes de almacenar en la sesión
        session['current_matrix'] = fraction_to_string(result)
        session['previous_op'] = 'multiply'
        session['original_a'] = fraction_to_string(matrix_a)
        session['original_b'] = fraction_to_string(matrix_b)
        return render_template("matrix_multiply.html", step=3, rows_a=rows_a, cols_a=cols_a, rows_b=rows_b, cols_b=cols_b, matrix_a=matrix_a, matrix_b=matrix_b, result=result, steps=steps)
    except ValueError as e:
        return render_template("matrix_multiply.html", step=2, rows_a=rows_a, cols_a=cols_a, rows_b=rows_b, cols_b=cols_b, matrix_a=matrix_a, matrix_b=matrix_b, error=f"Error: Ingrese valores válidos (admite fracciones tipo 3/2). {str(e)}")
    except Exception as e:
        return render_template("matrix_multiply.html", step=2, rows_a=rows_a, cols_a=cols_a, rows_b=rows_b, cols_b=cols_b, matrix_a=matrix_a, matrix_b=matrix_b, error=f"Error al realizar la multiplicación: {str(e)}")
    
@app.route("/matrix_transpose", methods=["GET", "POST"])
def matrix_transpose():
    use_result = request.args.get('use_result')
    if use_result and 'current_matrix' in session:
        matrix_a_str = session['current_matrix']  # Strings de la sesión
        matrix_a = string_to_fraction(matrix_a_str)  # ✅ Convertir a Fraction
        rows = len(matrix_a)
        cols = len(matrix_a[0])
        
        result, steps = ArOperations.transposeMatrixWithSteps(matrix_a)  # Ahora funciona
        
        # Lógica de verificación (necesita conversión también)
        verification = None
        if 'previous_op' in session:
            if session['previous_op'] == 'add':
                orig_a = string_to_fraction(session['original_a'])
                orig_b = string_to_fraction(session['original_b'])
                a_t, _ = ArOperations.transposeMatrixWithSteps(orig_a)
                b_t, _ = ArOperations.transposeMatrixWithSteps(orig_b)
                sum_t, _ = ArOperations.addTwoMatrixWithSteps(a_t, b_t)
                if fraction_to_string(sum_t) == fraction_to_string(result):
                    verification = "Propiedad verificada: (A + B)^T = A^T + B^T"
                else:
                    verification = "Propiedad diferente de: (A + B)^T != A^T + B^T"
            elif session['previous_op'] == 'subtract':
                orig_a = string_to_fraction(session['original_a'])
                orig_b = string_to_fraction(session['original_b'])
                a_t, _ = ArOperations.transposeMatrixWithSteps(orig_a)
                b_t, _ = ArOperations.transposeMatrixWithSteps(orig_b)
                sub_t, _ = ArOperations.subtractTwoMatrixWithSteps(a_t, b_t)
                if fraction_to_string(sub_t) == fraction_to_string(result):
                    verification = "Propiedad verificada: (A - B)^T = A^T - B^T"
                else:
                    verification = "Propiedad diferente de: (A - B)^T != A^T - B^T"
            elif session['previous_op'] == 'scalar':
                orig_a = string_to_fraction(session['original_a'])
                k = Fraction(session['original_scalar'])
                a_t, _ = ArOperations.transposeMatrixWithSteps(orig_a)
                kt, _ = ArOperations.multiplyMatrixByScalarWithSteps(a_t, k)
                if fraction_to_string(kt) == fraction_to_string(result):
                    verification = f"Propiedad verificada: ({str(k)} A)^T = {str(k)} A^T"
                else:
                    verification = f"Propiedad diferente de: ({str(k)} A)^T != {str(k)} A^T"
            elif session['previous_op'] == 'multiply':
                orig_a = string_to_fraction(session['original_a'])
                orig_b = string_to_fraction(session['original_b'])
                a_t, _ = ArOperations.transposeMatrixWithSteps(orig_a)
                b_t, _ = ArOperations.transposeMatrixWithSteps(orig_b)
                prod_t, _ = ArOperations.multiplyTwoMatrixWithSteps(b_t, a_t)
                if fraction_to_string(prod_t) == fraction_to_string(result):
                    verification = "Propiedad verificada: (A B)^T = B^T A^T"
                else:
                    verification = "Propiedad diferente de: (A B)^T != B^T A^T"
        
        session['current_matrix'] = fraction_to_string(result)
        session['previous_op'] = 'transpose'
        return render_template("matrix_transpose.html", step=3, rows=rows, cols=cols, 
                             matrix_a=matrix_a_str, result=result, steps=steps, 
                             verification=verification)
    
    # ... resto del código sin cambios

@app.route("/matrix_transpose_solve", methods=["POST"])
def matrix_transpose_solve():
    try:
        rows = int(request.form["rows"])
        cols = int(request.form["cols"])
        matrix_a = [[safe_fraction(request.form.get(f"A_{i}_{j}", 0)) for j in range(cols)] for i in range(rows)]
        result, steps = ArOperations.transposeMatrixWithSteps(matrix_a)
        session['current_matrix'] = fraction_to_string(result)
        session['previous_op'] = 'transpose'
        session['original_a'] = fraction_to_string(matrix_a)
        # Add verification logic here as before, using fraction_to_string for comparisons
        verification = None
        if 'previous_op' in session:
            if session['previous_op'] == 'add':
                orig_a = string_to_fraction(session['original_a'])
                orig_b = string_to_fraction(session['original_b'])
                a_t, _ = ArOperations.transposeMatrixWithSteps(orig_a)
                b_t, _ = ArOperations.transposeMatrixWithSteps(orig_b)
                sum_t, _ = ArOperations.addTwoMatrixWithSteps(a_t, b_t)
                if fraction_to_string(sum_t) == fraction_to_string(result):
                    verification = "Propiedad verificada: (A + B)^T = A^T + B^T"
                else:
                    verification = "Propiedad diferente de: (A + B)^T != A^T + B^T"
            elif session['previous_op'] == 'subtract':
                a_t, _ = ArOperations.transposeMatrixWithSteps(session['original_a'])
                b_t, _ = ArOperations.transposeMatrixWithSteps(session['original_b'])
                sub_t, _ = ArOperations.subtractTwoMatrixWithSteps(a_t, b_t)
                if fraction_to_string(sub_t) == fraction_to_string(result):
                    verification = "Propiedad verificada: (A - B)^T = A^T - B^T"
                else:
                    verification = "Propiedad diferente de: (A - B)^T != A^T - B^T"
            elif session['previous_op'] == 'scalar':
                a_t, _ = ArOperations.transposeMatrixWithSteps(session['original_a'])
                k = Fraction(session['original_scalar'])  # Convert back to Fraction for calculation
                kt, _ = ArOperations.multiplyMatrixByScalarWithSteps(a_t, k)
                if fraction_to_string(kt) == fraction_to_string(result):
                    verification = f"Propiedad verificada: ({str(k)} A)^T = {str(k)} A^T"
                else:
                    verification = f"Propiedad diferente de: ({str(k)} A)^T != {str(k)} A^T"
            elif session['previous_op'] == 'multiply':
                a_t, _ = ArOperations.transposeMatrixWithSteps(session['original_a'])
                b_t, _ = ArOperations.transposeMatrixWithSteps(session['original_b'])
                prod_t, _ = ArOperations.multiplyTwoMatrixWithSteps(b_t, a_t)
                if fraction_to_string(prod_t) == fraction_to_string(result):
                    verification = "Propiedad verificada: (A B)^T = B^T A^T"
                else:
                    verification = "Propiedad diferente de: (A B)^T != B^T A^T"
        return render_template("matrix_transpose.html", step=3, rows=rows, cols=cols, matrix_a=matrix_a, result=result, steps=steps, verification=verification)
    except ValueError as e:
        return render_template("matrix_transpose.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, error=f"Error: Ingrese valores válidos (admite fracciones tipo 3/2). {str(e)}")
    except Exception as e:
        return render_template("matrix_transpose.html", step=2, rows=rows, cols=cols, matrix_a=matrix_a, error=f"Error al realizar la transposición: {str(e)}")

# ========= Inversa por Gauss-Jordan =========
@app.route("/matrix_inverse", methods=["GET", "POST"])
def matrix_inverse():
    if request.method == "GET":
        return render_template("matrix_inverse.html", step=1)
    # POST (step 1) → pide n y pasa a step 2
    try:
        n = int(request.form.get("n"))
        if not (1 <= n <= 10):
            raise ValueError()
    except:
        return render_template("matrix_inverse.html", step=1, error="Dimensión inválida (1–10).")
    return render_template("matrix_inverse.html", step=2, n=n)

@app.route("/matrix_inverse_solve", methods=["POST"])
def matrix_inverse_solve():
    try:
        n = int(request.form.get("n"))
        A = [[safe_fraction(request.form.get(f"A_{i}_{j}")) for j in range(n)] for i in range(n)]
    except Exception:
        return render_template("matrix_inverse.html", step=2, n=request.form.get("n"),
                               error="Entrada inválida. Acepta enteros, decimales o fracciones a/b.")

    ok, steps, left, right, pivots, rank, reason = gauss_jordan_with_steps(A)
    props = props_invertibilidad(n, rank)

    if not ok:
        return render_template(
            "matrix_inverse_result.html",
            invertible=False, reason=reason, n=n, A=A,
            steps=steps, rank=rank, pivots=pivots, props=props
        )

    Ainv = right
    return render_template(
        "matrix_inverse_result.html",
        invertible=True, n=n, A=A, Ainv=Ainv,
        steps=steps, rank=rank, pivots=pivots, props=props
    )


def fraction_to_string(matrix):
    if isinstance(matrix, list):
        return [fraction_to_string(row) for row in matrix]
    elif isinstance(matrix, Fraction):
        return str(matrix)  # Convert Fraction to string (e.g., "3/2")
    return matrix

def string_to_fraction(matrix_str):
    """Convertir matriz de strings de vuelta a objetos Fraction"""
    if isinstance(matrix_str, list):
        return [string_to_fraction(row) for row in matrix_str]
    elif isinstance(matrix_str, str):
        try:
            return Fraction(matrix_str)
        except ValueError:
            try:
                return Fraction(float(matrix_str))
            except:
                return Fraction(0)
    return matrix_str

if __name__ == "__main__":
    app.run(debug=True)