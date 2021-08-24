import cv2

# гиперпараметры модели
conf_thr = 0.4
nms_thr = 0.6

# цвет ограничивающих прямоугольников и шрифта
color = (0, 255, 255)

# загрузка модели
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
with open('coco.names', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416))

# подключение к веб-камере
cam = cv2.VideoCapture(0)

while True:

    # чтение кадра
    ret_val, img = cam.read()

    # детектирование
    classes, scores, boxes = model.detect(img, conf_thr, nms_thr)

    # рисование прямоугольников, меток и вероятностей
    for class_id, score, box in zip(classes, scores, boxes):
        cv2.rectangle(img, box, color, 2)
        label = labels[class_id[0]]
        text = '{} {:.3f}'.format(label, score[0])
        coord = (box[0] + 5, box[1] + 15)
        cv2.putText(img, text, coord,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    # вывод на экрана, проверка нажатия клавиши Esc
    cv2.imshow('object detection', img)
    if cv2.waitKey(1) == 27: 
        break

