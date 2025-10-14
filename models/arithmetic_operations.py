class ArOperations:
    @staticmethod
    def addTwoMatrix(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Las matrices deben tener las mismas dimensiones para la suma.")
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] + matrix2[i][j])
            result.append(row)
        return result
    
    @staticmethod
    def subtractTwoMatrix(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Las matrices deben tener las mismas dimensiones para la resta.")
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] - matrix2[i][j])
            result.append(row)
        return result
    
    @staticmethod
    def multiplyMatrixByScalar(matrix, scalar):
        result = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[0])):
                row.append(matrix[i][j] * scalar)
            result.append(row)
        return result
    
    @staticmethod
    def multiplyTwoMatrix(matrix1, matrix2):
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("El numero de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz para la multiplicación.")
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix2[0])):
                sum_product = 0
                for k in range(len(matrix1[0])):
                    sum_product += matrix1[i][k] * matrix2[k][j]
                row.append(sum_product)
            result.append(row)
        return result
    
    @staticmethod
    def transposeMatrix(matrix):
        result = []
        for j in range(len(matrix[0])):
            row = []
            for i in range(len(matrix)):
                row.append(matrix[i][j])
            result.append(row)
        return result