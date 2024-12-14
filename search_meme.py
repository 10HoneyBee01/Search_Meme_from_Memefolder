import os
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

# Initialize CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Function to extract text features from the query text
def extract_text_features(query_text):
    inputs = processor(text=query_text, return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    return outputs

# Function to extract image features from meme images
def extract_image_features(meme_image, query_text):
    inputs = processor(text=query_text, images=meme_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs.image_embeds

# Function to query memes based on text similarity
def query_memes_by_text(query_text, meme_folder, top_n=5, similarity_weight=0.5):
    query_text_features = extract_text_features(query_text)
    
    image_features = []
    meme_files = []
    
    for filename in os.listdir(meme_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            meme_path = os.path.join(meme_folder, filename)
            meme_image = Image.open(meme_path)
            meme_image_features = extract_image_features(meme_image, query_text)
            image_features.append(meme_image_features)
            meme_files.append(filename)
    
    image_features = torch.stack(image_features).squeeze(1)
    text_similarities = cosine_similarity(query_text_features.detach().numpy(), image_features.detach().numpy()).flatten()
    image_similarities = cosine_similarity(query_text_features.detach().numpy(), image_features.detach().numpy()).flatten()

    combined_similarities = (similarity_weight * text_similarities + (1 - similarity_weight) * image_similarities)
    top_indices = combined_similarities.argsort()[-top_n:][::-1]
    top_memes = [(meme_files[i], combined_similarities[i], os.path.join(meme_folder, meme_files[i])) for i in top_indices]

    return top_memes

# Threaded function to avoid UI blocking
def on_query_meme_search(query_text, meme_folder, result_frame):
    try:
        results = query_memes_by_text(query_text, meme_folder)
        display_results(results, result_frame)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to update the GUI with the results
def display_results(results, result_frame):
    for widget in result_frame.winfo_children():
        widget.destroy()

    for i, (meme, score, meme_path) in enumerate(results, start=1):
        # Display Meme Image
        meme_image = Image.open(meme_path).resize((120, 120))  # Resize image for display
        meme_image_tk = ImageTk.PhotoImage(meme_image)

        # Add numbering and image to the result frame
        number_label = tk.Label(result_frame, text=f"{i}", font=("Helvetica", 14, "bold"))
        number_label.grid(row=i, column=0, padx=10, pady=10)

        image_label = tk.Label(result_frame, image=meme_image_tk)
        image_label.image = meme_image_tk  # Keep a reference to the image object
        image_label.grid(row=i, column=1, padx=10, pady=10)

        # Add name and similarity score
        label = tk.Label(result_frame, text=f"{meme}\nSimilarity: {score:.4f}", font=("Helvetica", 10))
        label.grid(row=i, column=2, padx=10, pady=10)

# Create the main window
root = tk.Tk()
root.title("Meme Query System")
root.geometry("900x700")

# Add input field for query text
query_label = tk.Label(root, text="Enter query text:", font=("Helvetica", 12))
query_label.pack(pady=10)

query_entry = tk.Entry(root, width=50, font=("Helvetica", 12))
query_entry.pack(pady=10)

# Function to browse for meme folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    folder_path.set(folder_selected)

# Add browse button for meme folder
folder_path = tk.StringVar()
folder_label = tk.Label(root, text="Select meme folder:", font=("Helvetica", 12))
folder_label.pack(pady=10)

folder_entry = tk.Entry(root, textvariable=folder_path, width=50, font=("Helvetica", 12))
folder_entry.pack(pady=10)

browse_button = tk.Button(root, text="Browse", command=browse_folder, font=("Helvetica", 12))
browse_button.pack(pady=5)

# Add a button to perform the meme search
def search_memes():
    query_text = query_entry.get()
    meme_folder = folder_path.get()
    if query_text and meme_folder:
        search_thread = Thread(target=on_query_meme_search, args=(query_text, meme_folder, result_frame))
        search_thread.start()
    else:
        messagebox.showwarning("Input Required", "Please enter a query and select a meme folder.")

search_button = tk.Button(root, text="Search Memes", command=search_memes, font=("Helvetica", 12))
search_button.pack(pady=10)

# Frame to display search results with images and names
result_frame = tk.Frame(root)
result_frame.pack(pady=20)

# Start the GUI
root.mainloop()
