import cv2
from deepface import DeepFace

# مقداردهی اولیه برای وضعیت قبلی احساس
previous_emotion = None

# باز کردن دوربین
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time emotion detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # تحلیل احساسات
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        # بررسی تغییر احساس
        if dominant_emotion != previous_emotion:
            print(f"🌀 تغییر احساس: از {previous_emotion} به {dominant_emotion}")
            previous_emotion = dominant_emotion

        # نوشتن احساس روی تصویر
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"[ERROR] {str(e)}")

    cv2.imshow('Emotion Watcher', frame)

    # خروج با کلید Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
