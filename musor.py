import telebot
import cv2
import numpy as np
import os
import torch
from moviepy.editor import VideoFileClip

BOT_TOKEN = 'BOT_TOKEN'
bot = telebot.TeleBot(BOT_TOKEN)

YOLO_WEIGHTS = 'C:\VisualCode\yolov4-tiny.weights'  
YOLO_CONFIG = 'C:\VisualCode\yolov4-tiny.cfg'      
YOLO_CLASSES = 'C:\VisualCode\coco.names.txt'     

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG) 

with open(YOLO_CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_objects(image):
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), font, 1, color, 2)

    return image




@bot.message_handler(commands=['start'])
def welcome(message):
    photo_path = 'C:\VisualCode\logo.png'

    try:
        with open(photo_path, 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption='Привет! 👋 Это бот команды "Ого-го-го". Я могу детектировать объекты на изображениях и видео с помощью модели YoLo. Просто отправь мне картинку(или же видео), и я покажу, что на ней изображено! 📷')
    except FileNotFoundError:
        bot.send_message(message.chat.id, "Ошибка: файл не найден.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.send_message(message.chat.id, "Спасибо за фото! Обрабатываю...")
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("received_image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    image = cv2.imread("received_image.jpg")

    processed_image = detect_objects(image)

    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, processed_image)

    with open(processed_image_path, 'rb') as img:
        bot.send_photo(message.chat.id, img)

def detect_objects_in_frame(frame):
    frame = frame.copy()

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            confidence = confidences[i]

            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def process_video(input_path, output_path):
    clip = VideoFileClip(input_path)
    processed_clip = clip.fl_image(detect_objects_in_frame)
    processed_clip.write_videofile(output_path, codec='libx264')

@bot.message_handler(content_types=['video'])
def handle_video(message):
    bot.send_message(message.chat.id, "Спасибо за видео! Обрабатываю...")
    
    file_info = bot.get_file(message.video.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    input_video_path = 'input_video.mp4'
    output_video_path = 'output_video.mp4'
    
    with open(input_video_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    
    process_video(input_video_path, output_video_path)

    with open(output_video_path, 'rb') as video:
        bot.send_video(message.chat.id, video)
    
    os.remove(input_video_path)
    os.remove(output_video_path)

    
bot.polling()
