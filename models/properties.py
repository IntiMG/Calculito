from fractions import Fraction

class Properties:
    def __init__(self, u, v, scalar, dimension, use_fractions=True):
        # conversión opcional a fracciones
        toF = (lambda x: Fraction(x).limit_denominator()) if use_fractions else (lambda x: x)
        self.u = [toF(x) for x in u]
        self.v = [toF(x) for x in v]
        self.scalar = toF(scalar)
        self.dimension = dimension
        self.use_fractions = use_fractions

        self.zero = [toF(0)] * dimension
        self.opposite_u = [-x for x in self.u]

    def sum_vectors(self, a, b):
        return [ai + bi for ai, bi in zip(a, b)]

    def scalar_mult(self, k, a):
        return [k * ai for ai in a]

    def get_computations(self):
        u, v, k = self.u, self.v, self.scalar
        sum_uv     = self.sum_vectors(u, v)
        sum_vu     = self.sum_vectors(v, u)
        k_u        = self.scalar_mult(k, u)
        sum_u_zero = self.sum_vectors(u, self.zero)
        sum_u_opp  = self.sum_vectors(u, self.opposite_u)

        # asociativa de la suma: tomamos w = k·v
        w          = self.scalar_mult(k, v)
        sum_uv_w   = self.sum_vectors(sum_uv, w)
        sum_u_vw   = self.sum_vectors(u, self.sum_vectors(v, w))

        return {
            'sum_uv': sum_uv,
            'sum_vu': sum_vu,
            'k_u': k_u,
            'zero': self.zero,
            'opposite_u': self.opposite_u,
            'sum_u_zero': sum_u_zero,
            'sum_u_opp': sum_u_opp,
            'w': w,
            'sum_uv_w': sum_uv_w,
            'sum_u_vw': sum_u_vw
        }

    def get_verifications(self):
        c = self.get_computations()
        return {
            'commutative':     c['sum_uv'] == c['sum_vu'],
            'associative':     c['sum_uv_w'] == c['sum_u_vw'],
            'zero_exists':     c['sum_u_zero'] == self.u,
            'opposite_exists': c['sum_u_opp'] == self.zero
        }

    def linear_combo_with_steps(vectors, coefs, use_fractions=True):
        """
        vectors: [[...], [...], ...]  (m vectores en R^n)
        coefs:   [a1, a2, ..., am]
        Devuelve: (resultado, pasos_dict)
        """
        toF = (lambda x: Fraction(x).limit_denominator()) if use_fractions else (lambda x: x)
        m = len(vectors)
        if m == 0:
            return [], {"scales": [], "adds": []}
        n = len(vectors[0])

        V = [[toF(x) for x in vec] for vec in vectors]
        A = [toF(a) for a in coefs]

        # 1) Escalar cada vector
        scales, scaled = [], []
        for j in range(m):
            sv = [A[j] * V[j][i] for i in range(n)]
            scaled.append(sv)
            scales.append({"a": A[j], "v": V[j], "res": sv})

        # 2) Sumas parciales acumuladas
        adds, current = [], [toF(0)] * n
        for j, sv in enumerate(scaled):
            new = [current[i] + sv[i] for i in range(n)]
            adds.append({"left": current, "right": sv, "res": new, "idx": j})
            current = new

        return current, {"scales": scales, "adds": adds}

    @staticmethod
    def mat_vec_with_steps(A, v, use_fractions=True):
        """
        Calcula Av y devuelve (resultado, pasos):
        - col_scales: cada v_j · (columna j de A)
        - col_adds: sumas parciales de esas columnas escaladas
        - row_dots: producto punto de cada fila de A con v (por componente)
        """
        from fractions import Fraction
        toF = (lambda x: Fraction(x).limit_denominator()) if use_fractions else (lambda x: x)

        if not A or not A[0]:
            return [], {"col_scales": [], "col_adds": [], "row_dots": []}
        m, n = len(A), len(A[0])
        if len(v) != n:
            raise ValueError("La longitud de v debe coincidir con el número de columnas de A.")

        # Convertir a fracciones
        A = [[toF(x) for x in row] for row in A]
        v = [toF(x) for x in v]

        # Columnas de A
        cols = [[A[i][j] for i in range(m)] for j in range(n)]

        # 1) Escalado de columnas por cada entrada de v
        col_scales = []
        scaled_cols = []
        for j in range(n):
            sj = [v[j] * cols[j][i] for i in range(m)]
            scaled_cols.append(sj)
            col_scales.append({"j": j, "scalar": v[j], "col": cols[j], "res": sj})

        # 2) Sumas parciales (acumuladas) de las columnas escaladas
        col_adds = []
        current = [toF(0)] * m
        for j, sj in enumerate(scaled_cols):
            new = [current[i] + sj[i] for i in range(m)]
            col_adds.append({"left": current, "right": sj, "res": new, "j": j})
            current = new
        result = current

        # 3) Cálculo por filas (producto punto fila_i · v)
        row_dots = []
        for i in range(m):
            terms = [A[i][j] * v[j] for j in range(n)]
            s = toF(0)
            partials = []
            for t in terms:
                s += t
                partials.append(s)
            row_dots.append({
                "i": i, "row": A[i], "terms": terms, "sum": s, "partials": partials
            })

        return result, {"col_scales": col_scales, "col_adds": col_adds, "row_dots": row_dots}