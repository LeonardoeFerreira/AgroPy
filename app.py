from ultralytics import YOLO
import cv2

# Carregar o modelo treinado
model = YOLO("runs/detect/.........pt")

# Função para analisar uma imagem com limiar de confiança reduzido
def analisar_folha(image_path, conf=0.25):  # Limiar de confiança = 0.25
    # Carregar a imagem
    image = cv2.imread(image_path)

    # Fazer a detecção
    results = model.predict(source=image, conf=conf)

    # Mostrar os resultados
    for result in results:
        boxes = result.boxes  # Caixas delimitadoras
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
            conf = box.conf[0]  # Confiança da detecção
            cls = box.cls[0]    # Classe detectada

            # Desenhar a bounding box na imagem
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Definir o texto da classe e confiança
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # Configurar a fonte e o tamanho do texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6   # Tamanho da fonte menor
            thickness = 2     # Espessura da fonte
            color = (0, 255, 0)  # Cor do texto (verde)
            
            # Escrever o texto na imagem
            cv2.putText(image, label, (x1, y1 - 10), font, font_scale, color, thickness)

    # Mostrar a imagem com as detecções
    cv2.imshow("Detecção de Doenças", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Testar com uma imagem
analisar_folha("C://Users//leofe//OneDrive//mancha_barrenta.png")
