import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def find_squares(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            squares.append(approx)

    return squares

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_distance_in_meters(distance_pixels, fator_de_escala):
    return distance_pixels * fator_de_escala

# Função para calcular a largura entre dois pontos
def calculate_width(point1, point2, fator_de_escala):
    distance_pixels = calculate_distance(point1, point2)
    distance_meters = calculate_distance_in_meters(distance_pixels, fator_de_escala)
    return distance_meters

def segment_acostamento_vegetacao(image, line1, line2):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Defina faixas de cor para acostamento (ajuste os valores conforme necessário)
    lower_acostamento = np.array([20, 50, 50], dtype=np.uint8)
    upper_acostamento = np.array([40, 255, 255], dtype=np.uint8)

    # Crie uma máscara para o acostamento
    mask_acostamento = cv2.inRange(hsv, lower_acostamento, upper_acostamento)

    # Aplique a máscara à imagem original
    acostamento = cv2.bitwise_and(image, image, mask=mask_acostamento)

    # Inverta a máscara para obter a vegetação
    mask_vegetacao = cv2.bitwise_not(mask_acostamento)
    vegetacao = cv2.bitwise_and(image, image, mask=mask_vegetacao)

    return acostamento, vegetacao

# Carrega a imagem a partir de um arquivo
image = cv2.imread('BR 421 KM 57 - KM 70 57.024 1.Jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height)
]

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)
cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi / 180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=25)

image_with_lines = draw_the_lines(image, lines)

# Encontra os quadrados na imagem
squares = find_squares(image)

# Desenha os quadrados na imagem original
cv2.drawContours(image_with_lines, squares, -1, (255, 0, 0), 2)

if len(lines) >= 2:
    line1 = lines[0][0]
    line2 = lines[1][0]
    point_faixa1 = (line1[0], line1[1])
    point_faixa2 = (line2[2], line2[3])

    # Adiciona uma linha horizontal azul da faixa até o canteiro de vegetação
    cv2.line(image_with_lines, (point_faixa1[0], point_faixa1[1]), (width, point_faixa1[1]), (255, 0, 0), thickness=2)

    # Calcula a distância entre os pontos extremos
    distance_pixels = calculate_distance(point_faixa1, (width, point_faixa1[1]))

    # Fator de escala para converter pixels para metros
    largura_real_da_faixa = 3.7  # em metros
    fator_de_escala = largura_real_da_faixa / distance_pixels

    # Adiciona uma linha horizontal azul da faixa até o acostamento
    cv2.line(image_with_lines, (point_faixa1[0], point_faixa1[1]), (line1[2], line1[3]), (255, 0, 0), thickness=2)

    # Calcula a distância horizontal entre a faixa e o acostamento
    distance_acostamento_pixels = calculate_distance((point_faixa1[0], point_faixa1[1]), (line1[2], line1[3]))
    if distance_acostamento_pixels != 0:  # Evita divisão por zero
        distance_acostamento_meters = calculate_distance_in_meters(distance_acostamento_pixels, fator_de_escala)

        # Adiciona o texto com a distância até o acostamento na imagem
        cv2.putText(image_with_lines, f'Distância até o acostamento: {distance_acostamento_meters:.2f} metros', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Calcula a largura do acostamento (distância entre as linhas)
        largura_acostamento_meters = calculate_width((line1[2], line1[3]), (line2[2], line2[3]), fator_de_escala)

        # Adiciona o texto com a largura do acostamento na imagem
        cv2.putText(image_with_lines, f'Largura do acostamento: {largura_acostamento_meters:.2f} metros', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Calcula a largura da faixa (distância entre a faixa e a linha 2)
        largura_faixa_meters = calculate_width((line2[2], line2[3]), (point_faixa1[0], point_faixa1[1]), fator_de_escala)

        # Adiciona o texto com a largura da faixa na imagem
        cv2.putText(image_with_lines, f'Largura da faixa: {largura_faixa_meters:.2f} metros', (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# ... (Códigos anteriores)

# Exibe a imagem resultante
plt.imshow(image_with_lines)
plt.show()