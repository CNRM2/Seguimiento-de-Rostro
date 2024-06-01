#------------------------------ Importamos las librerias ------------------------------
import cv2
import mediapipe as mp
from deepface import DeepFace

#------------------------------ Declaramos el detector --------------------------------
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

#------------------------------ Realizamos VideoCaptura --------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Inicializamos las variables globales
marca = 0
xmo, ymo = 0, 0

# --------------------------------- Funcion Mouse --------------------------------------
def mouse(evento, xm, ym, bandera, param):
    global xmo, ymo, marca
    # Evento doble click
    if evento == cv2.EVENT_LBUTTONDOWN:
        xmo = xm
        ymo = ym
        marca = 1
        print(xmo, ymo)

#-------------------------------Empezamos el while True --------------------------------
with detector.FaceDetection(min_detection_confidence=0.75, model_selection=0) as rostros:
    while True:
        # Lectura de fotogramas
        ret, frame = cap.read()

        # Aplicamos espejo a los frames
        frame = cv2.flip(frame, 1)

        # Correccion de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectamos los rostros
        resultado = rostros.process(rgb)

        # Creamos listas
        listacentro = []
        click = []
        listarostro = []

        # Si hay rostros entramos al if
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=(0,255,0),))

                for id, puntos in enumerate(resultado.detections):
                    # Mostramos toda la informacion
                    #print("Puntos: ", resultado.detections)

                    # Extraemos el ancho y el alto del frame
                    al, an, c = frame.shape

                    # Extraemos el medio de la pantalla
                    centro = int(an / 2)

                    # Extraemos las coordenadas X e Y min
                    x = puntos.location_data.relative_bounding_box.xmin
                    y = puntos.location_data.relative_bounding_box.ymin

                    # Extraemos el ancho y el alto
                    ancho = puntos.location_data.relative_bounding_box.width
                    alto = puntos.location_data.relative_bounding_box.height

                    # Pasamos X e Y a coordenadas en pixeles
                    x, y = int(x * an), int(y * al)
                    #print("X, Y: ", x, y)

                    # Pasamos el ancho y el alto a pixeles
                    x1, y1 = int(ancho * an), int(alto * al)
                    xf, yf = x + x1, y + y1

                    # Extraemos el punto central
                    cx = (x + (x + x1)) // 2
                    cy = (y + (y + y1)) // 2

                    listacentro.append([id, cx, cy])
                    listarostro.append([x, y, x1, y1])

                    # Mostrar un punto en el centro
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
                    cv2.line(frame, (cx, 0), (cx, 480), (0, 0, 255), 2)

                    cv2.namedWindow('Camara')
                    cv2.setMouseCallback('Camara', mouse)

                    # Marca
                    if marca == 1:
                        # SI estamos dentro de las coordenadas
                        if x < xmo < xf and y < ymo < yf:
                            # Dibujamos el click
                            cv2.circle(frame, (xmo, ymo), 20, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y), (xf, yf), (255, 255, 0), 3)  # Dibujamos el rectangulo
                            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                            xmo = cx
                            ymo = cy

                            print(resultado.detections[id])

                            # Extraemos la región del rostro para la detección de emociones
                            rostro_img = frame[y:yf, x:xf]
                            if rostro_img.size != 0:
                                try:
                                    # Detectamos la emoción
                                    analisis = DeepFace.analyze(rostro_img, actions=['emotion'], enforce_detection=False)
                                    emocion = analisis[0]['dominant_emotion'] if isinstance(analisis, list) else analisis['dominant_emotion']

                                    # Mostramos la emoción en la pantalla
                                    cv2.putText(frame, emocion, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    print(f"Emoción detectada: {emocion}")
                                except Exception as e:
                                    print(f"Error en la detección de emociones: {e}")

        cv2.imshow('Camara', frame)

        t = cv2.waitKey(1)
        if t == 27:
            break

cap.release()
cv2.destroyAllWindows()