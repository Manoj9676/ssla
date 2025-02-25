from tkinter import messagebox, simpledialog, filedialog, Text, Scrollbar, Button, Label, Tk, END, CENTER, W
import cv2 as cv
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tkinter

class TrafficSignLaneDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("SSLA Based Traffic Sign and Lane Detection for Autonomous Cars")
        master.geometry("1300x1200")
        master.config(bg='magenta3')
        
        self.filename = None
        self.model = None
        self.old = None
        self.class_labels = [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
            'No passing', 'Stop', 'No Entry', 'General caution', 'Traffic signals'
        ]

        self.setup_ui()
    
    def setup_ui(self):
        font = ('times', 16, 'bold')
        title = Label(self.master, text='SSLA Based Traffic Sign and Lane Detection for Autonomous cars', anchor=W, justify=CENTER)
        title.config(bg='yellow4', fg='white', font=font, height=3, width=120)
        title.place(x=0, y=5)

        font1 = ('times', 14, 'bold')
        upload_button = Button(self.master, text="Generate & Load Machine Learning Model", command=self.load_model)
        upload_button.place(x=50, y=100)
        upload_button.config(font=font1)

        self.pathlabel = Label(self.master, bg='yellow4', fg='white', font=font1)
        self.pathlabel.place(x=50, y=150)

        detect_button = Button(self.master, text="Upload Video & Detect Hough Lane, Signal", command=self.detect_signal)
        detect_button.place(x=50, y=200)
        detect_button.config(font=font1)

        exit_button = Button(self.master, text="Exit", command=self.master.quit)
        exit_button.place(x=50, y=250)
        exit_button.config(font=font1)

        font2 = ('times', 12, 'bold')
        self.text = Text(self.master, height=15, width=78)
        scroll = Scrollbar(self.text)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.place(x=450, y=100)
        self.text.config(font=font2)

    def canny_detection(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
        canny_img = cv.Canny(blur_img, 50, 150)
        return canny_img

    def segment_detection(self, img):
        height = img.shape[0]
        polygons = np.array([[(0, height), (800, height), (380, 290)]])
        mask_img = np.zeros_like(img)
        cv.fillPoly(mask_img, polygons, 255)
        segment_img = cv.bitwise_and(img, mask_img)
        return segment_img

    def calculate_lines(self, frame, lines):
        left, right = [], []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, y_intercept = parameters
            (left if slope < 0 else right).append((slope, y_intercept))
        left_avg = np.average(left, axis=0) if left else None
        right_avg = np.average(right, axis=0) if right else None
        return np.array([
            self.calculate_coordinates(frame, left_avg) if left_avg is not None else None,
            self.calculate_coordinates(frame, right_avg) if right_avg is not None else None
        ])

    def calculate_coordinates(self, frame, parameters):
        if parameters is None:
            return None
        slope, intercept = parameters
        y1 = frame.shape[0]
        y2 = int(y1 - 150)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def visualize_lines(self, frame, lines):
        lines_visualize = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                if line is not None:
                    x1, y1, x2, y2 = line
                    cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
        return lines_visualize

    def load_model(self):
        try:
            self.model = load_model('model/model.h5')
            self.pathlabel.config(text="Machine Learning Traffic Sign Detection Model Loaded")
            self.text.delete('1.0', END)
            self.text.insert(END, "Machine Learning Traffic Sign Detection Model Loaded\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def detect_signal(self):
        filename = filedialog.askopenfilename(initialdir="Videos")
        if not filename:
            return
        self.pathlabel.config(text=filename)
        self.text.delete('1.0', END)
        self.text.insert(END, f"{filename} loaded\n\n")
        self.text.update_idletasks()

        cap = cv.VideoCapture(filename)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            canny = self.canny_detection(frame)
            cv.imshow("Canny Image", canny)
            segment = self.segment_detection(canny)
            hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
            if hough is not None:
                lines = self.calculate_lines(frame, hough)
                lines_visualize = self.visualize_lines(frame, lines)
                output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
            else:
                output = frame.copy()
            
            cv.imwrite("test.jpg", output)
            temps = cv.imread("test.jpg")
            h, w, c = temps.shape
            image = load_img("test.jpg", target_size=(80, 80))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            (box_preds, label_preds) = self.model.predict(image)
            box_preds = box_preds[0]
            startX = int(box_preds[0] * w)
            startY = int(box_preds[1] * h)
            endX = int(box_preds[2] * w)
            endY = int(box_preds[3] * h)
            predict = np.argmax(label_preds, axis=1)[0]
            accuracy = np.amax(label_preds, axis=1)[0]
            if accuracy > 0.97:
                cv.putText(output, f"Recognized As {self.class_labels[predict]}", (startX, startY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.imshow("Output", output)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    app = TrafficSignLaneDetectionApp(root)
    root.mainloop()
