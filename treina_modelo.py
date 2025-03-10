
from ultralytics import YOLO

if __name__ == '__main__':
    # Carregue o modelo YOLOv8l pré-treinado
    model = YOLO("yolov8l.pt")  # Carrega o modelo YOLOv8 Large

    # Continue o treinamento na GPU
    results = model.train(
        data="C://Users//leofe//OneDrive//Documentos//Python//AgroPy//dataset.yaml",  # Caminho para o dataset.yaml    
        epochs=100,            # Número de épocas
        imgsz=640,            # Tamanho da imagem
        batch=4,              # Tamanho do batch
        lr0=0.001,            # Taxa de aprendizado
        name="peanut_leaf3",  # Nome do treino
        device=0,             # Usar a GPU (índice 0)
        workers=0             # Desativa o multiprocessamento (recomendado para Windows)
    )
