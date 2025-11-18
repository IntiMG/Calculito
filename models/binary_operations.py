import math
#notacion cientifica y tmb n decimales, paso a paso y mostrar la formula
#funciones exponenciales y logaritmicas y trigonometricas polinomiales4
#propagacion de errores derivando. integrales
#graficas 
#log comun

# Función auxiliar para convertir a binario con procedimiento de divisiones+
def _dec_to_bin_procedure(decimal, bits):
    """Convierte un decimal a binario (parte entera) con procedimiento detallado."""
    decimal = abs(decimal)
    if decimal == 0:
        return '0' * bits, "<p>El valor decimal es 0. La representación binaria es simplemente 0.</p>"

    procedimiento = "<h4>Procedimiento: División Sucesiva por 2</h4>"
    procedimiento += "<table class='solution-table'><thead><tr><th>Cociente</th><th>Resto (Bit)</th></tr></thead><tbody>"
    
    bin_list = []
    cociente = decimal
    
    while cociente > 0:
        resto = cociente % 2
        procedimiento += f"<tr><td>{cociente} / 2 = {cociente // 2}</td><td>{resto}</td></tr>"
        bin_list.append(str(resto))
        cociente //= 2

    binario = "".join(reversed(bin_list))
    
    if len(binario) > bits:
        procedimiento += f"</tbody></table><p class='error'>ADVERTENCIA: Se necesitan {len(binario)} bits, pero solo se especificaron {bits}. Se utilizarán {len(binario)} bits.</p>"
        bits = len(binario)

    padding = '0' * (bits - len(binario))
    binario_pad = padding + binario
    
    procedimiento += f"</tbody></table>"
    procedimiento += f"<p>Leyendo los restos de abajo hacia arriba: El binario es <code>{binario}</code>.</p>"
    procedimiento += f"<p>Ajustando a {bits} bits (con padding de ceros a la izquierda): <code>{binario_pad}</code></p>"
    
    return binario_pad, procedimiento

# ----------------------------------------------------------------------
# 1. Decimal a Binario (Principal)
# ----------------------------------------------------------------------

def dec_to_bin(decimal_str, bits_str):
    """Calcula la conversión Decimal a Binario y devuelve el procedimiento."""
    try:
        decimal = int(decimal_str)
        bits = int(bits_str)
        if bits <= 0:
            return None, "<p class='error'>El número de bits debe ser positivo.</p>", "Error"
    except ValueError:
        return None, "<p class='error'>Por favor, introduzca valores numéricos válidos.</p>", "Error"

    # Conversion
    binario, procedimiento = _dec_to_bin_procedure(decimal, bits)
    
    resultado_texto = f"El decimal {decimal} en binario (utilizando {bits} bits) es: <code>{binario}</code>"
    
    return binario, procedimiento, resultado_texto

# ----------------------------------------------------------------------
# 2. Hexadecimal a Binario y Viceversa
# ----------------------------------------------------------------------

def get_hex_conversion_table():
    """Genera la tabla de conversión Hex/Dec/Bin 0-15."""
    html_table = "<h4>Tabla de Conversión Base 10, Base 2, y Base 16</h4>"
    html_table += "<table class='solution-table' style='text-align:center;'><thead><tr><th>Decimal (Base 10)</th><th>Binario (Base 2)</th><th>Hexadecimal (Base 16)</th></tr></thead><tbody>"
    
    for i in range(16):
        binario = f"{i:04b}" # 4 bits
        hexadecimal = f"{i:X}"
        html_table += f"<tr><td>{i}</td><td>{binario}</td><td>{hexadecimal}</td></tr>"
        
    html_table += "</tbody></table>"
    return html_table

def hex_to_bin_proc(hex_str):
    """Convierte Hex a Binario, mostrando la sustitución de 4 bits."""
    hex_str = hex_str.upper().strip()
    procedimiento = "<h4>Procedimiento: Sustitución de 4 bits</h4>"
    procedimiento += get_hex_conversion_table()
    procedimiento += f"<p>Separamos cada dígito hexadecimal de <code>{hex_str}</code> y lo reemplazamos por su equivalente binario de 4 bits:</p>"
    
    binario_completo = ""
    for char in hex_str:
        try:
            dec_val = float(char, 16)
            bin_val = f"{dec_val:04b}"
            procedimiento += f"<li>Dígito Hex: <code>{char}</code> &rarr; Binario: <code>{bin_val}</code></li>"
            binario_completo += bin_val
        except ValueError:
            return None, f"<p class='error'>Carácter Hexadecimal no válido: {char}</p>", "Error"

    procedimiento += f"<p>Concatenando los bloques: <code>{binario_completo}</code></p>"
    resultado_texto = f"El valor Hexadecimal {hex_str} en binario es: <code>{binario_completo}</code>"
    return binario_completo, procedimiento, resultado_texto

# ----------------------------------------------------------------------
# 3. Signed, Unsigned, Complemento a 2
# ----------------------------------------------------------------------

def get_signed_c2(decimal_str, bits_str):
    """Calcula y muestra los procedimientos de representación de números enteros."""
    try:
        decimal = int(decimal_str)
        bits = int(bits_str)
        if bits <= 1:
            return None, "<p class='error'>Se requieren al menos 2 bits.</p>", "Error"
    except ValueError:
        return None, "<p class='error'>Por favor, introduzca valores numéricos válidos.</p>", "Error"

    html_content = "<h2>Representaciones de Número Entero</h2>"
    
    # 3.1. Rango de Valores
    unsigned_max = 2**bits - 1
    # Signed Magnitude range
    signed_max_mag = 2**(bits - 1) - 1
    c2_min = -(2**(bits - 1))
    c2_max = 2**(bits - 1) - 1
    
    html_content += "<div class='solution-box'>"
    html_content += f"<h3>Información de Rango para {bits} Bits</h3>"
    html_content += "<table class='solution-table'><thead><tr><th>Representación</th><th>Valor Mínimo</th><th>Valor Máximo</th></tr></thead><tbody>"
    html_content += f"<tr><td>Sin Signo (Unsigned)</td><td>0</td><td>{unsigned_max}</td></tr>"
    html_content += f"<tr><td>Signo y Magnitud (Signed)</td><td>{-signed_max_mag}</td><td>{signed_max_mag}</td></tr>"
    html_content += f"<tr><td>Complemento a 2 (C2)</td><td>{c2_min}</td><td>{c2_max}</td></tr>"
    html_content += "</tbody></table></div>"

    if decimal < c2_min or decimal > unsigned_max:
        html_content += f"<p class='error'>El valor decimal ({decimal}) está fuera del rango máximo sin signo ({unsigned_max}) y mínimo de Complemento a 2 ({c2_min}) para {bits} bits. No se puede representar.</p>"
        return None, html_content, "Rango Excedido"

    # 3.2. Representación Sin Signo (Unsigned)
    html_content += "<div class='solution-box'>"
    html_content += "<h3>1. Sin Signo (Unsigned)</h3>"
    if decimal < 0:
        html_content += f"<p class='error'>El número ({decimal}) es negativo. La representación Sin Signo solo funciona para números $\\geq 0$.</p>"
        unsigned_bin = "N/A"
    elif decimal > unsigned_max:
         html_content += f"<p class='error'>El número ({decimal}) excede el máximo sin signo ({unsigned_max}) para {bits} bits.</p>"
         unsigned_bin = "Rango Excedido"
    else:
        unsigned_bin, proc_u = _dec_to_bin_procedure(decimal, bits)
        html_content += proc_u
        html_content += f"<p>Representación Final: <code>{unsigned_bin}</code></p>"
    html_content += "</div>"
    
    # 3.3. Representación Signo y Magnitud
    html_content += "<div class='solution-box'>"
    html_content += "<h3>2. Signo y Magnitud (Signed)</h3>"
    
    if decimal < -signed_max_mag or decimal > signed_max_mag:
        html_content += f"<p class='error'>El número ({decimal}) está fuera del rango de Signo y Magnitud (Mín: {-signed_max_mag}, Máx: {signed_max_mag}) para {bits} bits.</p>"
        signed_bin = "Rango Excedido"
    else:
        sign_bit = '0' if decimal >= 0 else '1'
        magnitud = abs(decimal)
        
        # Binario de la magnitud (N-1 bits, dejando 1 para el signo)
        bin_magnitude = bin(magnitud)[2:].zfill(bits - 1)
        signed_bin = sign_bit + bin_magnitude
        
        html_content += f"<p>Paso 1: Bit de Signo. ({'Positivo' if decimal >= 0 else 'Negativo'}) $\\rightarrow$ <code>{sign_bit}</code></p>"
        html_content += f"<p>Paso 2: Binario de la magnitud ({magnitud}) en {bits-1} bits $\\rightarrow$ <code>{bin_magnitude}</code></p>"
        html_content += f"<p>Representación Final: <code>{signed_bin}</code></p>"
        
    html_content += "</div>"


    # 3.4. Representación Complemento a 2 (C2)
    html_content += "<div class='solution-box'>"
    html_content += "<h3>3. Complemento a 2 (C2)</h3>"
    
    if decimal >= 0:
        # Positivo o Cero: es el mismo binario
        c2_bin, proc_abs = _dec_to_bin_procedure(decimal, bits)
        html_content += proc_abs
        html_content += f"<p>El número es positivo, por lo que la representación C2 es el binario de su magnitud, ajustado a {bits} bits.</p>"
    else:
        # Negativo: C2
        magnitud = abs(decimal)
        
        # Paso 1: Binario de la magnitud
        bin_n_bits = bin(magnitud)[2:].zfill(bits)
        
        html_content += "<h4>Paso 1: Binario de la magnitud ($|x|$) en " + str(bits) + " bits</h4>"
        html_content += f"<p>Binario de la magnitud ({magnitud}): <code>{bin_n_bits}</code></p>"
        
        # Paso 2: Complemento a 1 (Inversión)
        comp_uno = ''.join(['1' if b == '0' else '0' for b in bin_n_bits])
        html_content += f"<h4>Paso 2: Complemento a 1 (Invertir bits)</h4>"
        html_content += f"<p>C1: <code>{comp_uno}</code></p>"
        
        # Paso 3: Complemento a 2 (Sumar 1)
        c2_int = int(comp_uno, 2) + 1
        c2_bin = bin(c2_int)[2:].zfill(bits)
        
        html_content += f"<h4>Paso 3: Complemento a 2 (Sumar 1)</h4>"
        html_content += f"<p>C1 + 1 = <code>{c2_bin}</code></p>"
        
    html_content += f"<p>Representación Final C2 de {decimal}: <code>{c2_bin}</code></p>"
    html_content += "</div>"
    
    return c2_bin, html_content, f"Representación C2 de {decimal} en {bits} bits: <code>{c2_bin}</code>"

# ----------------------------------------------------------------------
# 4. Punto Flotante (IEEE-754 conceptual) - CORREGIDO
# ----------------------------------------------------------------------

def floating_point(decimal_str, bits_sign_str, bits_exponent_str, bits_mantissa_str):
    """Calcula la representación en Punto Flotante con procedimiento detallado."""
    try:
        decimal = float(decimal_str)
        bits_s = int(bits_sign_str)
        bits_e = int(bits_exponent_str)
        bits_m = int(bits_mantissa_str)
        total_bits = bits_s + bits_e + bits_m
        
        if bits_s != 1 or bits_e < 3 or bits_m < 3:
             return None, f"<p class='error'>El bit de signo debe ser 1. El exponente y la mantisa deben ser $\\geq 3$ para un cálculo razonable.</p>", "Error"
    except ValueError:
        return None, "<p class='error'>Por favor, introduzca valores numéricos válidos para el decimal y los bits.</p>", "Error"
    
    if decimal == 0.0:
        cero_float = '0' * total_bits
        html_content = f"<h2>Representación Punto Flotante (IEEE-754 Conceptual)</h2>"
        html_content += f"<p class='matrix-info'>El valor es $0.0$. La representación es toda a cero: <code>{cero_float}</code></p>"
        return cero_float, html_content, f"0.0 en FP ({total_bits} bits): <code>{cero_float}</code>"

    html_content = f"<h2>Representación Punto Flotante (IEEE-754 Conceptual)</h2>"
    html_content += f"<p class='matrix-info'>Total de Bits: {total_bits} (Signo: {bits_s} | Exponente: {bits_e} | Mantisa: {bits_m})</p>"
    
    # 1. Signo
    sign_bit = '0' if decimal >= 0 else '1'
    html_content += "<div class='solution-box'>"
    html_content += "<h3>Paso 1: Bit de Signo (S)</h3>"
    html_content += f"<p>Valor decimal: {decimal}. Bit de Signo: <code>{sign_bit}</code></p>"
    html_content += "</div>"
    
    magnitud = abs(decimal)
    
    # 2. Conversión a Binario y Normalización
    html_content += "<div class='solution-box'>"
    html_content += "<h3>Paso 2: Conversión a Binario y Normalización</h3>"
    
    entera = int(magnitud)
    fraccionaria = magnitud - entera
    
    # Convertir parte entera
    bin_entera = bin(entera)[2:]
    
    # Convertir parte fraccionaria (Aumentado a 50 bits para mayor precisión)
    bin_fraccionaria = ""
    temp_frac = fraccionaria
    proc_frac = "<h4>Conversión Fraccionaria: Multiplicación Sucesiva por 2</h4>"
    proc_frac += "<table class='solution-table'><thead><tr><th>Fracción Inicial</th><th>x 2</th><th>Parte Entera (Bit)</th></tr></thead><tbody>"
    
    # Intentamos obtener suficientes bits para normalizar y truncar
    for _ in range(bits_m + 15): # Se aumentó el límite de iteraciones
        if temp_frac == 0:
            break
        result = temp_frac * 2
        entera_bit = int(result)
        temp_frac = result - entera_bit
        
        # Redondeo para mostrar en el procedimiento (la lógica usa el float puro)
        proc_frac += f"<tr><td>{temp_frac/2 if _ > 0 else fraccionaria:.10f}</td><td>{result:.10f}</td><td>{entera_bit}</td></tr>"
        bin_fraccionaria += str(entera_bit)

    proc_frac += "</tbody></table>"
    html_content += proc_frac
    
    binario_completo = bin_entera + "." + bin_fraccionaria
    html_content += f"<p>Binario no normalizado (aproximado): <code>{binario_completo}</code></p>"
    
    # Normalización: 1.XXXXX * 2^E
    
    if entera > 0:
        # Número >= 1 (Ej: 101.101 -> 1.01101 * 2^2)
        E = len(bin_entera) - 1
        # La mantisa es la parte que sigue al primer '1'
        mantissa_proc = bin_entera[1:] + bin_fraccionaria
    else:
        # Número < 1 (Ej: 0.00101 -> 1.01 * 2^-3)
        match_one = bin_fraccionaria.find('1')
        if match_one == -1: 
            error_msg = "<p class='error'>Error de Normalización: El número es demasiado pequeño, lo que lleva a un exponente fuera de rango (subnormal/cero).</p>"
            return None, html_content + error_msg, "Error de Normalización (Subnormal)"
        
        E = -(match_one + 1)
        # Nos saltamos los ceros iniciales y el primer '1' implícito
        mantissa_proc = bin_fraccionaria[match_one + 1:]
    
    # Extraer la mantisa requerida (bits_m)
    mantissa_trunc = mantissa_proc[:bits_m]
    
    # Rellenar Mantisa con ceros si es necesario 
    mantissa = mantissa_trunc.ljust(bits_m, '0')
    
    html_content += f"<h4>Normalización</h4>"
    html_content += f"<p>Forma normalizada: $1.\\texttt{{{mantissa_proc[:bits_m+4]}...}} \\times 2^{{{E}}}$</p>"
    html_content += f"<p>Exponente real (E): {E}</p>"
    html_content += f"<p>Mantisa (M) (sin el '1' implícito), truncada a {bits_m} bits: <code>{mantissa}</code></p>"
    html_content += "</div>"
    
    # 3. Exponente con Sesgo (Bias)
    html_content += "<div class='solution-box'>"
    html_content += "<h3>Paso 3: Exponente Sesgado (E')</h3>"
    
    bias = 2**(bits_e - 1) - 1
    exp_sesgado_dec = E + bias
    
    html_content += f"<p>Fórmula: $E' = E + Bias$</p>"
    html_content += f"<p>Bias para {bits_e} bits: $2^{{{bits_e}-1}} - 1 = {bias}$</p>"
    html_content += f"<p>Exponente Sesgado (Decimal): ${E} + {bias} = {exp_sesgado_dec}$</p>"

    # Rango del exponente sesgado
    E_min_dec = 1 # Para desnormalizados
    E_max_dec = 2**bits_e - 2 # Para el máximo finito
    
    if exp_sesgado_dec < E_min_dec or exp_sesgado_dec > E_max_dec:
        error_msg = f"<h4>Error de Exponente</h4><p class='error'>El exponente sesgado ({exp_sesgado_dec}) está fuera del rango normal. El resultado es Subnormal, Infinito o NaN.</p>"
        html_content += error_msg
        exp_sesgado_bin = '1' * bits_e 
        
        resultado_final = sign_bit + exp_sesgado_bin + mantissa
        return resultado_final, html_content, f"Flotante de {decimal}: RANGO EXCEDIDO (Inf/NaN)"

    exp_sesgado_bin = bin(exp_sesgado_dec)[2:].zfill(bits_e)
    html_content += f"<p>Exponente Sesgado (Binario de {bits_e} bits): <code>{exp_sesgado_bin}</code></p>"
    html_content += "</div>"
    
    # 4. Resultado Final
    resultado_final = sign_bit + exp_sesgado_bin + mantissa
    html_content += "<div class='solution-box' style='background:#e6ffed; border-color:#3fa37a;'>"
    html_content += "<h3>Paso 4: Ensamblaje Final</h3>"
    html_content += "<p>Concatenando [S] [E'] [M]:</p>"
    html_content += f"<pre style='font-size:1.1em; border:none; background:none;'>[<code>{sign_bit}</code>] [<code>{exp_sesgado_bin}</code>] [<code>{mantissa}</code>]</pre>"
    html_content += f"<p>Representación Final de {decimal}: <code>{resultado_final}</code></p>"
    html_content += "</div>"

    return resultado_final, html_content, f"Flotante de {decimal} en Base 2 ({total_bits} bits): <code>{resultado_final}</code>"