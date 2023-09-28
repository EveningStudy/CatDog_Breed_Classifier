import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ExifTags
from utils_ import classification_main_onnxruntime
import threading


class AboutWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("About CatDog Breed Classifier")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = screen_width // 3
        window_height = screen_height // 3
        self.root.geometry(f"{window_width + 100}x{window_height}")

        self.label = tk.Label(self.root, text="About CatDog Breed Classifier", font=("Helvetica", 16))
        self.label.pack(pady=20)

        about_text = """
        Author: Chris Wang
        Contact: s22019.wang@stu.scie.com.cn
        Repository: https://github.com/EveningStudy/CatDog_Breed_Classifier
        """

        self.about_label = tk.Label(self.root, text=about_text, font=("Helvetica", 12), justify="left")
        self.about_label.pack(padx=20, pady=10)


class CatDogClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CatDog Breed Classifier")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg="#f2f2f2")

        self.create_widgets()
        self.running = False

    def create_widgets(self):
        self.label = tk.Label(self.root, text="CatDog Breed Classifier", font=("Helvetica", 28), bg="#f2f2f2")
        self.label.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Upload an image:", font=("Helvetica", 14), bg="#f2f2f2")
        self.image_label.pack()

        self.image_frame = tk.Frame(self.root, bg="#f2f2f2")
        self.image_frame.pack()

        self.image_path = tk.StringVar()
        self.image_path_entry = tk.Entry(self.image_frame, textvariable=self.image_path, width=50,
                                         font=("Helvetica", 12))
        self.image_path_entry.pack(side=tk.LEFT, padx=5, pady=10)

        self.browse_button = tk.Button(self.image_frame, text="Browse", command=self.browse_image,
                                       font=("Helvetica", 12), relief=tk.RAISED)
        self.browse_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_input,
                                      font=("Helvetica", 16), bg="#FF5722", fg="white", relief=tk.RAISED)
        self.clear_button.pack(pady=20)

        self.run_button = tk.Button(self.root, text="Run Classifier", command=self.run_classifier,
                                    font=("Helvetica", 16), bg="#4CAF50", fg="white", relief=tk.RAISED)
        self.run_button.pack(pady=20)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 18), bg="#f2f2f2")
        self.result_label.pack()

        self.image_display = tk.Label(self.root, bg="#f2f2f2")
        self.image_display.pack()

        self.author_label = tk.Label(self.root, text="Made by: Chris Wang", font=("Helvetica", 12), bg="#f2f2f2")
        self.author_label.pack(side=tk.BOTTOM, pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image_path.set(file_path)
            self.display_image(file_path)

    def display_image(self, file_path, category=None, top5_results=None):
        image = Image.open(file_path)

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        image.thumbnail((400, 400))
        image = ImageTk.PhotoImage(image)

        self.image_display.configure(image=image)
        self.image_display.image = image

    def run_classifier(self):
        if self.running:
            return

        image_path = self.image_path.get()
        if image_path == '0':
            exit(0)

        try:
            if not image_path:
                raise ValueError("Please select an image.")
            if not Image.open(image_path):
                raise ValueError("File not found.")
        except Exception as e:
            self.display_error_message(str(e))
            return

        self.running = True
        self.run_button.config(state=tk.DISABLED, text="Running...")
        self.result_label.config(text="Running Classifier...", fg="#4CAF50")

        thread = threading.Thread(target=self.run_classifier_thread, args=(image_path,))
        thread.start()

    def run_classifier_thread(self, image_path):
        preds, output = classification_main_onnxruntime.cat_and_dog_classification(image_path)
        if preds[0] < 0.97:
            self.result_label.config(text="The image is neither a dog nor a cat.", fg="#F44336")
        elif output:
            preds_dog = classification_main_onnxruntime.dog_classification(image_path)
            self.display_top5_results(preds_dog, "Dog")
        elif output == 0:
            preds_cat = classification_main_onnxruntime.cat_classification(image_path)
            self.display_top5_results(preds_cat, "Cat")

        self.running = False
        self.run_button.config(state=tk.NORMAL, text="Run Classifier")
        self.result_label.update()

    def display_top5_results(self, predictions, category):
        self.result_label.config(text=f"{category} (Top 5 Results):", fg="#4CAF50")
        result_text = f"{category} (Top 5 Results):\n"
        for i, (class_name, prob) in enumerate(predictions[:5], start=1):
            result_text += f"{i}. {class_name}: {prob * 100:.2f}%\n"
        self.result_label.update()
        self.result_label.config(text=result_text)

    def display_error_message(self, error_message):
        messagebox.showerror("Error", error_message)

    def clear_input(self):
        self.image_path.set("")
        self.image_display.configure(image=None)
        self.result_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogClassifierGUI(root)


    def open_about_window():
        about_window = tk.Toplevel(root)
        about_app = AboutWindow(about_window)


    about_button = tk.Button(root, text="About", command=open_about_window, font=("Helvetica", 12), relief=tk.RAISED)
    about_button.pack(pady=10)

    root.mainloop()
