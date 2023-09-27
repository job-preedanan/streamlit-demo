import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from DINOv2 import DinoVisionTransformerClassifier as dino_classifier
import gdown


# Define the class names
class_names = ['Normal', 'APP', 'EP', 'Irrelevant']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource()
def download_weights(url):
    gdown.download(url, "DINOb_518_best.pt", quiet=False)


# @st.cache_re(allow_output_mutation=True)
@st.cache_resource()
def load_model(model_path):
    model = dino_classifier(len(class_names), model_size='b').to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    # model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model (assuming it's a saved PyTorch model)
    model.eval()
    return model


@st.cache_data
def plot_probabilities(probabilities):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, probabilities, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Class Probabilities')

    return fig


@st.cache_data
def load_image(image):
    return Image.open(image)


# @st.cache_data
def classify_image(image, model, image_size, multi_label=False):
    # Define the image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the image to the specified size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])

    # Preprocess the input image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(input_batch)

    # If multi_label is True, use sigmoid to get probabilities
    if multi_label:
        # Get the predicted class probabilities
        probabilities = torch.sigmoid(output[0])
        predicted_class = (probabilities.data > 0.5).float()      # multi outputs
    else:
        # Get the predicted class probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the class index with the highest probability
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_class_idx]        # 1 output

    return predicted_class, probabilities.tolist()


def main():
    st.title('Swine Diseases Classification Demo')

    # Create sidebar for user input
    st.sidebar.header('User Input')
    image = st.sidebar.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    image_size = 518
    # model_path = st.sidebar.file_uploader('Upload the model file (PyTorch .pt)', type=['pt'])
    model_url = 'https://drive.google.com/uc?export=download&id=1o14U3yNxIBQPU5dD86IPjs8o5FffuFfw'
    download_weights(model_url)
    model_path = 'DINOb_518_best.pt'
    multi_label = st.sidebar.checkbox('Multi-output Classification', value=True)

    if image is not None and model_path is not None:
        model = load_model(model_path)
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load and classify the uploaded image
        image = load_image(image)
        # image = Image.open(image)

        # Use st.beta_columns to create a layout
        col1, col2 = st.columns([1, 1])

        # Display the image in the first column
        col1.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_classes, probabilities = classify_image(image, model, image_size, multi_label)
        print(predicted_classes, probabilities)

        if multi_label:
            # Display the results
            result_list = []
            for i in range(len(class_names)):
                if predicted_classes[i] == 1:
                    result_list.append(class_names[i])

            result = '+'.join(result_list)
            st.header('Result :: ' + str(result))
        else:
            st.header('Result :: ' + str(predicted_classes))

        # Display the bar graph in the second column
        with col2:
            st.write('Class Probabilities:')
            barplot = plot_probabilities(probabilities)
            st.pyplot(barplot)


if __name__ == '__main__':
    main()
