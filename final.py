import cv2 # type: ignore
import numpy as np  # type: ignore
import mediapipe as mp # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from PIL import ImageFont, ImageDraw, Image # type: ignore
import tkinter as tk
from tkinter import Label

# تحميل النموذج المدرب
model = load_model('asl_model_improved.h5')
# إعداد Mediapipe لاكتشاف معالم اليد
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# التأكد من فتح الكاميرا بنجاح
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# القاموس الذي يربط بين الحروف الإنجليزية والعربية
english_to_arabic = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'هـ', 'haa': 'ح',
    'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ط',
    'taa': 'ت', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ئ',
    'yaa': 'ي', 'zay': 'ز'
}

# دالة لاستخراج معالم اليد من الإطار
def extract_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            # استخراج حدود اليد لرسم مستطيل حولها
            h, w, _ = frame.shape
            xmin = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            ymin = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            xmax = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            ymax = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
            return landmarks, (xmin, ymin, xmax, ymax)
    return None, None

# دالة لاستخراج الحرف المتنبأ به
def get_predicted_letter(frame):
    hand_landmarks, bbox = extract_hand_landmarks(frame)
    if hand_landmarks is not None:
        landmarks = np.array(hand_landmarks).flatten().reshape(1, -1)
        pred = model.predict(landmarks)
        
        # حساب أعلى وأقل احتمال
        max_pred = np.max(pred)
        if max_pred > 0.90:  # إذا كانت النسبة أكبر من 90% يمكننا الثبات على التنبؤ
            predicted_class = np.argmax(pred, axis=1)
            predicted_label = list(english_to_arabic.keys())[predicted_class[0]]
            return english_to_arabic[predicted_label], bbox, max_pred
        else:
            return None, None, max_pred
    return None, None, 0

# إعداد نافذة Tkinter لعرض الحرف المتنبأ به
def create_prediction_window():
    window = tk.Tk()
    window.title("Predicted Letter")
    window.geometry("300x200")
    label = Label(window, text="Predicted: ", font=("Arial", 20))
    label.pack(expand=True)

    def update_prediction(new_prediction):
        label.config(text=f"Predicted: {new_prediction}")

    return window, update_prediction

# إعداد نافذة Tkinter لعرض الحرف المتنبأ به
def create_prediction_window():
    window = tk.Tk()
    window.title("Predicted Letter")
    window.geometry("400x200")
    label = Label(window, text="Predicted: ", font=("Arial", 20))
    label.pack(expand=True)

    def update_prediction(new_prediction, confidence):
        if confidence > 0.90:
            label.config(
                text=f"Predicted: {new_prediction}\nConfidence: {confidence*100:.2f}%",
                fg="green"
            )
        else:
            label.config(
                text=f"Predicted: {new_prediction}\nConfidence: {confidence*100:.2f}%",
                fg="red"
            )

    return window, update_prediction

# إنشاء نافذة Tkinter
window, update_prediction = create_prediction_window()

# متغير لتخزين آخر حرف تم التنبؤ به
last_predicted_letter = None

while True:
    # قراءة الإطار من الكاميرا
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # الحصول على الحرف المتنبأ به
    predicted_letter, bbox, max_pred = get_predicted_letter(frame)

    # تحديث نافذة Tkinter بالحرف المتنبأ
    if predicted_letter:
        update_prediction(predicted_letter, max_pred)
    
    # إذا كانت اليد موجودة، ارسم مستطيل حولها
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # عرض الإطار الناتج مع التنبؤ
    cv2.imshow('Sign Language Recognition', frame)

    # الخروج من الحلقة إذا ضغطت على مفتاح "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # تحديث نافذة Tkinter
    window.update()

# تحرير الكاميرا وإغلاق جميع النوافذ المفتوحة
cap.release()
cv2.destroyAllWindows()
