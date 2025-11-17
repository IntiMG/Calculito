# models/positional_notation.py

class PositionalNotation:
    """
    Módulo para descomponer números en notación posicional
    en base 10 y base 2, mostrando TODOS los pasos que pide el profesor.
    """

    @staticmethod
    def descomponer_base_10(numero: int):
        """
        Descompone un número entero en base 10 usando potencias de 10.

        Retorna:
          - expresion_suma: string tipo "8 × 10^4 + 4 × 10^3 + ..."
          - pasos: lista de strings con cada multiplicación
          - resultado: el número original (comprobación de la suma)
        """
        if numero < 0:
            raise ValueError("Solo se permiten números enteros no negativos para esta descomposición.")

        numero_str = str(numero)
        longitud = len(numero_str)

        pasos = []
        terminos = []

        for i, digito_char in enumerate(numero_str):
            # posición desde la izquierda → potencia desde la derecha
            digito = int(digito_char)
            potencia = longitud - i - 1
            valor_posicional = 10 ** potencia
            producto = digito * valor_posicional

            # "mostrar la multiplicación de cada cifra por su valor posicional"
            pasos.append(
                f"{digito} × 10^{potencia} = {digito} × {valor_posicional} = {producto}"
            )

            # término tipo "8 × 10^4"
            terminos.append(f"{digito} × 10^{potencia}")

        # "mostrar la suma final"
        expresion_suma = " + ".join(terminos) + f" = {numero}"

        return expresion_suma, pasos, numero

    @staticmethod
    def descomponer_base_2(binario: str):
        """
        Descompone un número en base 2 (cadena de 0s y 1s) usando potencias de 2.

        Retorna:
          - expresion_suma: string tipo "1 × 2^6 + 1 × 2^5 + ..."
          - pasos: lista de strings con cada multiplicación
          - resultado_decimal: valor en base 10 de ese binario
        """
        # Validar que solo contenga 0 y 1
        if not binario or any(c not in "01" for c in binario):
            raise ValueError("El número en base 2 debe contener solo dígitos 0 y 1.")

        longitud = len(binario)

        pasos = []
        terminos = []
        resultado_decimal = 0

        for i, bit_char in enumerate(binario):
            bit = int(bit_char)
            potencia = longitud - i - 1
            valor_posicional = 2 ** potencia
            producto = bit * valor_posicional

            # "mostrar el valor de cada dígito multiplicado por su correspondiente potencia de 2"
            pasos.append(
                f"{bit} × 2^{potencia} = {bit} × {valor_posicional} = {producto}"
            )

            terminos.append(f"{bit} × 2^{potencia}")
            resultado_decimal += producto

        # "mostrar la suma final"
        expresion_suma = " + ".join(terminos) + f" = {resultado_decimal}"

        return expresion_suma, pasos, resultado_decimal


# Pequeña prueba manual (solo si ejecutas este archivo directamente)
if __name__ == "__main__":
    # Ejemplo en base 10: 84506
    expr10, pasos10, res10 = PositionalNotation.descomponer_base_10(84506)
    print("Descomposición en base 10:")
    for p in pasos10:
        print("  ", p)
    print("  Suma final:", expr10)
    print()

    # Ejemplo en base 2: 1111001
    expr2, pasos2, res2 = PositionalNotation.descomponer_base_2("1111001")
    print("Descomposición en base 2:")
    for p in pasos2:
        print("  ", p)
    print("  Suma final:", expr2, f"(= {res2} en base 10)")
