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
class_names_4cls = ['Normal', 'APP', 'EP', 'Irrelevant']
class_names_3cls = ['Normal', 'Diseases', 'Irrelevant']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource()
def download_weights(url, model_name):
    gdown.download(url, model_name, quiet=False)


# @st.cache_re(allow_output_mutation=True)
@st.cache_resource()
def load_model(model_paths, class_num):
    if class_num == 3:
        model_path = model_paths[0]
        model = dino_classifier(len(class_names_3cls), model_size='b').to(device)
    elif class_num == 4:
        model_path = model_paths[1]
        model = dino_classifier(len(class_names_4cls), model_size='b').to(device)
    print('model detail - ')
    print(str(class_num), model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    # model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model (assuming it's a saved PyTorch model)
    model.eval()
    return model


@st.cache_data
def plot_probabilities(probabilities, num_classes):
    if num_classes == 3:
        class_names = class_names_3cls
    elif num_classes == 4:
        class_names = class_names_4cls

    fig, ax = plt.subplots()
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, probabilities, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Class Probabilities')

    return fig


# def post_process(outputs, th):
#     if (outputs > 0.5).sum() == 0:
#         predicted = torch.zeros_like(outputs)
#         predicted[outputs == torch.max(outputs)] = 1
#         return predicted
#
#     else:
#         predicted = (outputs.data > th).float()
#         # print(predicted)
#         if predicted[0] > 0.5 and (predicted[1] > 0.5 or predicted[2] > 0.5):
#             predicted[0] = 0
#
#         return predicted


def post_process(outputs, th_normal=0.3, th_disease=0.3):
    # this function processes the sigmoid output from model to predict the final result

    # If there is no class that prob > 0.5  --> return class with maximum prob
    if (outputs > th_disease).sum() == 0:
        predicted = torch.zeros_like(outputs)
        predicted[outputs == torch.max(outputs)] = 1
        return predicted

    else:
        # apply thresholds
        predicted = torch.zeros_like(outputs)
        predicted[0] = 1 if outputs[0] > th_normal else 0
        predicted[1] = 1 if outputs[1] > th_disease else 0
        predicted[2] = 1 if outputs[2] > th_disease else 0
        predicted[3] = 1 if outputs[3] > th_normal else 0

        # case that Normal appears together with APP or EP --> remove Normal prediction, showing only diseases
        if predicted[0] == 1 and (predicted[1] == 1 or predicted[2] == 1):
            predicted[0] = 0

        # case that Irrelevant appears together with other classes --> remove Irrelevant prediction, showing only diseases
        if predicted[3] == 1 and (predicted[1] == 1 or predicted[2] == 1):
            predicted[3] = 0

        return predicted

@st.cache_data
def load_image(image):
    return Image.open(image)


# @st.cache_data
def classify_image(image, model, image_size, num_classes=4):
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

    # If multi_label is True, use sigmoid to get probabilities (class_num =4)
    if num_classes == 4:
        # Get the predicted class probabilities
        probabilities = torch.sigmoid(output[0])
        # predicted_class = (probabilities.data > 0.5).float()      # multi outputs
        predicted_class = post_process(probabilities, 0.3, 0.3)            # post processing

    elif num_classes == 3:
        # Get the predicted class probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the class index with the highest probability
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = class_names_3cls[predicted_class_idx]        # 1 output

    return predicted_class, probabilities.tolist()


def main():
    st.title('Swine Diseases Classification Demo')

    # Create sidebar for user input
    st.sidebar.header('User Input')
    image = st.sidebar.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    image_size = 336

    # Add a checkbox to select the multi-label (possible multiple output) or multi-class (only the highest prob.)
    # multi_label = st.sidebar.checkbox('Multi-output Classification', value=True)

    # Add a slider to select the number of classes ()
    num_classes = st.sidebar.slider("Normal/Diseases or Normal/APP/EP", min_value=3, max_value=4)

    # load models
    # model_path = st.sidebar.file_uploader('Upload the model file (PyTorch .pt)', type=['pt'])
    # model_url = 'https://drive.google.com/uc?export=download&id=1o14U3yNxIBQPU5dD86IPjs8o5FffuFfw' # old 518
    # model_url_3cls = 'https://drive.google.com/uc?export=download&id=1_8gX7MKa02i7sIgjJ1pJEuGzpfouHPl-'
    model_url_3cls = 'https://drive.google.com/uc?export=download&id=1XQTyOh-wZ98Sb1XtyUVwenXwvlM-LQOo'
    # model_url_4cls = 'https://drive.google.com/uc?export=download&id=16fLZFDg7_lrMdYV57GuzL78IWoAHpVKl'   # 336 multi-label
    # model_url_4cls = 'https://drive.google.com/uc?export=download&id=1PbAE97wdXWnNSYmEJ_hiNk3j84ybOoud'
    model_url_4cls = 'https://drive.google.com/uc?export=download&id=1A95ZG6OxhEVJibQiNm2_-BCC3L-SZJJq'
    model_path_3cls = 'DINOb(f)_3cls_336_4_best.pt'
    model_path_4cls = 'DINOb_4cls_336_best.pt'
    if num_classes == 3:
        download_weights(model_url_3cls, model_path_3cls)
    elif num_classes==4:
        download_weights(model_url_4cls, model_path_4cls)
    model_paths = [model_path_3cls, model_path_4cls]

    if image is not None and model_paths is not None:
        model = load_model(model_paths, num_classes)
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load and classify the uploaded image
        image = load_image(image)

        # Use st.beta_columns to create a layout
        col1, col2 = st.columns([1, 1])

        # Display the image in the first column
        col1.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_classes, probabilities = classify_image(image, model, image_size, num_classes)
        print(predicted_classes, probabilities)

        # 4 classes classification (multi-label output)
        if num_classes == 4:
            # Display the results
            result_list = []
            for i in range(len(class_names_4cls)):
                if predicted_classes[i] == 1:
                    result_list.append(class_names_4cls[i])

            result = '+'.join(result_list)
            st.header('Result :: ' + str(result))

        # 3 classes classifcation (multi-class - single output)
        elif num_classes == 3:
            st.header('Result :: ' + str(predicted_classes))

        # Display the bar graph in the second column
        with col2:
            st.write('Class Probabilities:')
            barplot = plot_probabilities(probabilities, num_classes)
            st.pyplot(barplot)


if __name__ == '__main__':
    main()
