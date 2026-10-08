# Jesus Arriaga 
# NC= 0019
import cv2
import numpy as np

# Cargar imagen
imagen = cv2.imread("cebra.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen 'cebra.jpg'. Verifica la ruta y el nombre.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar los puntos detectados
esquinas = cv2.dilate(
    esquinas,
    None
)

# ---------------------------------------------------------
# RESULTADO 1: Umbral Normal (Esquinas Principales - 1%)
# ---------------------------------------------------------
resultado1 = imagen.copy()
umbral1 = 0.01 * esquinas.max()
resultado1[esquinas > umbral1] = [0, 0, 255]  # Marcar en Rojo

# ---------------------------------------------------------
# RESULTADO 2: Umbral Sensible (Todas las Esquinas/Detalles - 0.1%)
# ---------------------------------------------------------
resultado2 = imagen.copy()
umbral2 = 0.001 * esquinas.max()
resultado2[esquinas > umbral2] = [0, 255, 0]  # Marcar en Verde

# Configurar dimensiones de ventanas escaladas
ancho_ventana = 800
alto_ventana = 600

# Mostrar Ventana 1
cv2.namedWindow("Resultado 1 - Esquinas Principales 0019", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resultado 1 - Esquinas Principales 0019", ancho_ventana, alto_ventana)
cv2.imshow("Resultado 1 - Esquinas Principales 0019", resultado1)

# Mostrar Ventana 2
cv2.namedWindow("Resultado 2 - Mayor Sensibilidad 0019", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resultado 2 - Mayor Sensibilidad 0019", ancho_ventana, alto_ventana)
cv2.imshow("Resultado 2 - Mayor Sensibilidad 0019", resultado2)

# Guardar ambos resultados
cv2.imwrite("../resultados/ejemplo2_cebra_resultado1.jpg", resultado1)
cv2.imwrite("../resultados/ejemplo2_cebra_resultado2.jpg", resultado2)

# Conteo de puntos
cant1 = np.sum(esquinas > umbral1)
cant2 = np.sum(esquinas > umbral2)

print("Detección de esquinas terminada.")
print("Puntos detectados en Resultado 1:", cant1)
print("Puntos detectados en Resultado 2:", cant2)
print("Resultados guardados en ../resultados/")

# Esperar tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
print("PROGRAMA REALIZADO POR JESUS ARRIAGA 0019")
