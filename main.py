import cv2

face_ref = cv2.CascadeClassifier("Wajah.xml")
smile_cascade = cv2.CascadeClassifier("Senyum.xml")
camera = cv2.VideoCapture(0)

def deteksi_wajah(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face_ref.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return wajah

def deteksi_senyum(roi_gray, x, y):
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_gray, (sx, sy), (sx + sw, sy + sh), (180, 0, 0), 2)
        cv2.putText(roi_gray, 'Smile', (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,0,0), 1)

def gambar_kotak(frame):
    for (x, y, w, h) in deteksi_wajah(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = frame[y:y+h, x:x+w]
        deteksi_senyum(roi_gray, x, y)
        

def main():
    while True:
        _, frame = camera.read()
        gambar_kotak(frame)
        cv2.imshow("contoh", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        
if __name__ == '__main__':
    main()
