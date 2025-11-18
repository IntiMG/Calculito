class ArOperations:
    @staticmethod
    def addTwoMatrixWithSteps(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Las matrices deben tener las mismas dimensiones para la suma.")
        result = []
        steps = []
        for i in range(len(matrix1)):
            row = []
            row_steps = []
            for j in range(len(matrix1[0])):
                sum_val = matrix1[i][j] + matrix2[i][j]
                row_steps.append(f"{str(matrix1[i][j])} + {str(matrix2[i][j])} = {str(sum_val)}")
                row.append(sum_val)
            steps.append(row_steps)
            result.append(row)
        return result, steps
    
    @staticmethod
    def subtractTwoMatrixWithSteps(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Las matrices deben tener las mismas dimensiones para la resta.")
        result = []
        steps = []
        for i in range(len(matrix1)):
            row = []
            row_steps = []
            for j in range(len(matrix1[0])):
                sub_val = matrix1[i][j] - matrix2[i][j]
                row_steps.append(f"{str(matrix1[i][j])} - {str(matrix2[i][j])} = {str(sub_val)}")
                row.append(sub_val)
            steps.append(row_steps)
            result.append(row)
        return result, steps
    
    @staticmethod
    def multiplyMatrixByScalarWithSteps(matrix, scalar):
        result = []
        steps = []
        for i in range(len(matrix)):
            row = []
            row_steps = []
            for j in range(len(matrix[0])):
                prod_val = matrix[i][j] * scalar
                row_steps.append(f"{str(scalar)} * {str(matrix[i][j])} = {str(prod_val)}")
                row.append(prod_val)
            steps.append(row_steps)
            result.append(row)
        return result, steps
    
    @staticmethod
    def multiplyTwoMatrixWithSteps(matrix1, matrix2):
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("El numero de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz para la multiplicación.")
        result = []
        steps = []
        for i in range(len(matrix1)):
            row = []
            row_steps = []
            for j in range(len(matrix2[0])):
                sum_product = 0
                term_steps = []
                for k in range(len(matrix1[0])):
                    prod = matrix1[i][k] * matrix2[k][j]
                    term_steps.append(f"{str(matrix1[i][k])} * {str(matrix2[k][j])}")
                    sum_product += prod
                row_steps.append(f"({' + '.join(term_steps)}) = {str(sum_product)}")
                row.append(sum_product)
            steps.append(row_steps)
            result.append(row)
        return result, steps
    
    @staticmethod
    def transposeMatrixWithSteps(matrix):
        result = []
        steps = []
        for j in range(len(matrix[0])):
            row = []
            row_steps = []
            for i in range(len(matrix)):
                val = matrix[i][j]
                row_steps.append(f"Posición original ({i+1},{j+1}) -> Transpuesta ({j+1},{i+1}): {str(val)}")
                row.append(val)
            steps.append(row_steps)
            result.append(row)
        return result, steps