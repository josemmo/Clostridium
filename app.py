import tkinter as tk
from tkinter import filedialog
import subprocess

def browse_file():
    global file_path
    file_path = filedialog.askopenfilename()
    dataset_label.config(text=file_path)

def browse_directory():
    global storing_path
    storing_path = filedialog.askdirectory()
    results_label.config(text=storing_path)

def preprocessing_apply():
    subprocess.call (["/usr/bin/Rscript", "--vanilla", "preprocess_maldi.R", file_path, storing_path])

def model1_apply():
    # Add code to run model 1 here
    pass

def model2_apply():
    # Add code to run model 2 here
    pass

# Create the main window
root = tk.Tk()
root.title("Python GUI")
root.geometry("500x500")

# Add a button to browse files
browse_button = tk.Button(root, text="Browse MALDI-TOF input", command=browse_file)
browse_button.pack()

# Add a label to display the selected file path
dataset_label = tk.Label(root)
dataset_label.pack()

# Add a button to browse directories
browse_directory_button = tk.Button(root, text="Select a folder to store results" , command=browse_directory)
browse_directory_button.pack()

# Add a label to display the selected results directory
results_label = tk.Label(root)
results_label.pack()

# Add a checkbox for preprocessing
preprocessing_var = tk.IntVar()
preprocessing_checkbox = tk.Checkbutton(root, text="Preprocessing", variable=preprocessing_var)
preprocessing_checkbox.pack()

# Add a button to run preprocessing
preprocessing_button = tk.Button(root, text="Apply", command=preprocessing_apply)
preprocessing_button.pack()

# Add checkboxes for models 1 and 2
model1_var = tk.IntVar()
model1_checkbox = tk.Checkbutton(root, text="Model 1", variable=model1_var)
model1_checkbox.pack()
model1_button = tk.Button(root, text="Apply", command=model1_apply)
model1_button.pack()

model2_var = tk.IntVar()
model2_checkbox = tk.Checkbutton(root, text="Model 2", variable=model2_var)
model2_checkbox.pack()
model2_button = tk.Button(root, text="Apply", command=model2_apply)
model2_button.pack()

# Run the main loop
root.mainloop()
