import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from solutions import helper  # Replace with your helper module


def load_image(label: tk.Label) -> Image:
    """
    Open a file dialog to load an image and display its filename.

    Args:
        label (tk.Label): The label to display the selected filename.

    Returns:
        Image: The loaded image.
    """
    filename = filedialog.askopenfilename()
    label.config(text=filename)
    return Image.open(filename)


def predict(
    image1_label: tk.Label,
    image2_label: tk.Label,
    panel1: tk.Label,
    panel2: tk.Label,
    result_label: tk.Label,
) -> None:
    """
    Load and preprocess two images, display them, and make a prediction.

    Args:
        image1_label (tk.Label): Label displaying the filename of the first image.
        image2_label (tk.Label): Label displaying the filename of the second image.
        panel1 (tk.Label): Label to display the first image.
        panel2 (tk.Label): Label to display the second image.
        result_label (tk.Label): Label to display the prediction result.

    Returns:
        None
    """
    image1, show_im1 = helper.load_and_preprocess_image(image1_label.cget("text"))
    image2, show_im2 = helper.load_and_preprocess_image(image2_label.cget("text"))

    show_im1 = show_im1.resize((250, 250), Image.ANTIALIAS)
    show_im2 = show_im2.resize((250, 250), Image.ANTIALIAS)
    img1 = ImageTk.PhotoImage(show_im1)
    img2 = ImageTk.PhotoImage(show_im2)
    panel1.config(image=img1)
    panel1.image = img1
    panel2.config(image=img2)
    panel2.image = img2

    # Replace this with your prediction logic
    result = helper.predict_function(image1, image2)
    result_label.config(text="Prediction: " + result)


def main() -> None:
    root = tk.Tk()

    root.geometry("800x600")  # Set window size

    image_frame = tk.Frame(root)
    image_frame.pack(pady=20)

    image1_label = tk.Label(root, text="")
    image1_button = tk.Button(
        root, text="Pick Image 1", command=lambda: load_image(image1_label)
    )
    image1_button.pack()

    image2_label = tk.Label(root, text="")
    image2_button = tk.Button(
        root, text="Pick Image 2", command=lambda: load_image(image2_label)
    )
    image2_button.pack()

    panel1 = tk.Label(image_frame)
    panel1.pack(side="left", padx=10)

    panel2 = tk.Label(image_frame)
    panel2.pack(side="left")

    predict_button = tk.Button(
        root,
        text="Predict",
        command=lambda: predict(
            image1_label, image2_label, panel1, panel2, result_label
        ),
    )
    predict_button.pack(pady=20)

    result_label = tk.Label(root, text="", font=("Helvetica", 16))
    result_label.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
