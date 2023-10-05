import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Inicialize a webcam
cap = cv2.VideoCapture(0)  # O argumento 0 indica a câmera padrão (geralmente a webcam embutida)

# Carrega modelo gerado no treinamento

    
modelo = load_model('./prediction.h5', compile=False)#

# Verifique se a câmera foi aberta corretamente
if not cap.isOpened():
    print('Erro ao abrir a câmera.')
    exit()

# Defina a largura e a altura do quadro de vídeo
largura = 640  
altura = 480

img_size = 145

# Defina a resolução do quadro de vídeo
cap.set(3, largura)
cap.set(4, altura)

# Loop para capturar imagens e vídeos da webcam
while True:
    # Leia um quadro da câmera
    ret, frame = cap.read()

    # Exiba o quadro na janela
    cv2.imshow('Webcam', frame)
    
    frame = cv2.resize(frame, (img_size, img_size)) # 145 mesmo valor usado no treinamento

    # Pré-processamento do quadro
    frame = frame / 255.0  # Normalize os valores de pixel para o intervalo [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Adicione uma dimensão de lote
    

    # Faça previsões com o modelo
    previsoes = modelo.predict(frame)
    
    print("previsão: ", previsoes)
    
    texto_classe = f'Classe: {previsoes}'
    cv2.putText(frame, texto_classe, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Libere o objeto de captura da câmera
cap.release()

# Feche todas as janelas
cv2.destroyAllWindows()