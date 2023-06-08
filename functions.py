import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from skimage import filters
from PIL import Image
import numpy as np
import plotly.express as px
import os
import base64
import shutil
import gdown


def modelsCheck():
    models_url_id = {
        'UNet' : '1kyUFiaLXwClJDPCq3U-ae1ZHamhEm7Z-',
        'UNet++' : '1t-Z3AltZ-Kr0TvgZJd-QwJ4Na6e9MCuW'
        }
    isModelsExist = []
    for model_name in models_url_id.keys():
        folder_path = 'models/{}/'.format(model_name)
        model_filename = model_name + '.h5'
        isModelExist = os.path.isfile(os.path.join(folder_path, model_filename))
        isModelsExist.append(isModelExist)
    
    if False in isModelsExist:
        # HAVE TO UPDATE THE URL IF MODEL UPDATED IN COLAB
        for name, model_url_id in zip(models_url_id.keys(), models_url_id.values()):
            save_path = 'models/{}/{}.h5'.format(name, name)
            url = "https://drive.google.com/uc?id={}".format(model_url_id)
            gdown.download(url, save_path, quiet=False)


def refreshImagesInFolder():
    folder_path = 'results_images'
    # Melakukan iterasi pada setiap file dalam folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

def createSegmentation(model, image_arr):
    # Image Array
    image_arr_reshape = image_arr[np.newaxis, ...] # Reshape menjadi (1, width, height, channel) agar bisa predict

    # Model Predict Tensorflow
    mask_arr = model.predict(image_arr_reshape, verbose=0)

    # Create Segmented Images
    segmented_image_arr = image_arr * mask_arr[0]

    # Create Line Segmentation
    line_mask_arr = filters.sobel(mask_arr[0])

    # Convert Image to uint8
    image_arr_uint8 = (image_arr*255).astype(np.uint8)
    line_mask_arr_uint8 = (line_mask_arr*255).astype(np.uint8)
    
    # Convert Image to PIL
    image_pil = Image.fromarray(image_arr_uint8[:, :, 0])
    line_mask_pil = Image.fromarray(line_mask_arr_uint8[:, :, 0])

    # Convert image to RGBA
    line_mask_pil_rgba = line_mask_pil.convert("RGBA")
    image_pil_rgba = image_pil.convert("RGBA")

    # Convert image to tuple data
    line_mask_pil_data = line_mask_pil_rgba.getdata()
    image_pil_data = image_pil_rgba.getdata()

    image_result = []
    for mask_line_channels, image_channels in zip(line_mask_pil_data, image_pil_data):
        if mask_line_channels != (0, 0, 0, 255):
            image_channels = list(image_channels)
            image_channels[0] = int((image_channels[0] + 180) / 2)
            image_channels[1] = int((image_channels[1] + 0) / 2)
            image_channels[2] = int((image_channels[2] + 0) / 2)
            # image_channels[3] = 120
            image_channels = tuple(image_channels)
            image_channels = image_channels
            image_result.append(image_channels)
        else:
            image_result.append(image_channels)
    image_pil_rgba.putdata(image_result)

    IMAGE = array_to_img(image_arr)
    MASK = array_to_img(mask_arr[0])
    LINE_MASK = image_pil_rgba
    SEGMENTED_IMAGE = array_to_img(segmented_image_arr)
    
    MERGE_IMAGE = Image.new('RGB', (256*3, 256))
    MERGE_IMAGE.paste(IMAGE, (0, 0))
    MERGE_IMAGE.paste(LINE_MASK, (256, 0))
    MERGE_IMAGE.paste(MASK, (256*2, 0))

    return IMAGE, MASK, LINE_MASK, MERGE_IMAGE, SEGMENTED_IMAGE

def getArrImages():
    images_arr = []
    if st.session_state.input_selected == 'camera':
        image_session = st.session_state.images
        image =  load_img(image_session, color_mode='grayscale')
        width, height = image.size
        # Set coordinat to crop image
        left = (width/2) - (height/2)
        top = 0
        right = (width/2) + (height/2)
        bottom = height
        # Cut image at center coordinate
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image_resize =  cropped_image.resize((256, 256))
        image_arr = img_to_array(cropped_image_resize).astype('float32') / 255.0
        images_arr.append(image_arr)
    else:
        images_session = st.session_state.images
        for image_session in images_session:
            image =  load_img(image_session, color_mode='grayscale', target_size=(256, 256))
            image_arr = img_to_array(image).astype('float32') / 255.0
            images_arr.append(image_arr)

    return np.array(images_arr)

def saveSegmentation():
    # Refresh or Delete All Images In Directory
    refreshImagesInFolder()
    # Get Images Array
    images_arr = getArrImages()
    # Get Used Model from Session
    used_model = st.session_state.used_model
    if used_model == 'All Models At Once':
        model_names = ['UNet', 'UNet++']
        for model_name in model_names:
            model = tf.keras.models.load_model('./models/{}/{}.h5'.format(model_name, model_name), compile=False)
            for i, image_arr in enumerate(images_arr):
                ori_image, mask_image, line_mask_image, merge_image, segmented_image = createSegmentation(model, image_arr)
                # Save Image to Directory
                _ = ori_image.save("results_images/{}/ori_images/".format(model_name) + "{}_ori_{}".format(model_name, i+1)+".bmp") # image
                _ = mask_image.save("results_images/{}/mask_images/".format(model_name) + "{}_mask_{}".format(model_name, i+1)+".bmp")  # mask
                _ = line_mask_image.save("results_images/{}/line_mask_images/".format(model_name) + "{}_line_{}".format(model_name, i+1)+".bmp")  # line_mask
                _ = merge_image.save("results_images/{}/merge_images/".format(model_name) + "{}_merge_{}".format(model_name, i+1)+".bmp")  # merge_image
                _ = segmented_image.save("results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp")  # merge_image
            
    else:
        model_name = used_model.split()[0]
        model = tf.keras.models.load_model('./models/{}/{}.h5'.format(model_name, model_name), compile=False)
        for i, image_arr in enumerate(images_arr):
            ori_image, mask_image, line_mask_image, merge_image, segmented_image = createSegmentation(model, image_arr)
            # Save Image to Directory
            _ = ori_image.save("results_images/{}/ori_images/".format(model_name) + "{}_ori_{}".format(model_name, i+1)+".bmp") # image
            _ = mask_image.save("results_images/{}/mask_images/".format(model_name) + "{}_mask_{}".format(model_name, i+1)+".bmp")  # mask
            _ = line_mask_image.save("results_images/{}/line_mask_images/".format(model_name) + "{}_line_{}".format(model_name, i+1)+".bmp")  # line_mask
            _ = merge_image.save("results_images/{}/merge_images/".format(model_name) + "{}_merge_{}".format(model_name, i+1)+".bmp")  # merge_image
            _ = segmented_image.save("results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp")  # segmented_image

def showImageMarkdown(path):
    image = open(path, "rb") 
    contents = image.read() 
    data_url = base64.b64encode(contents).decode("utf-8") 
    image.close()
    st.markdown( f'<p style="text-align: center;"><img src="data:image/gif;base64,{data_url}" alt="results_images"></p>', unsafe_allow_html=True)

def showSegmentationFromCamera():
    # Get Used Model from Session
    used_model = st.session_state.used_model
    # Process All Models At Once
    if used_model == 'All Models At Once':
        model_names = ['UNet', 'UNet++']
        for model_name in model_names:
            # MODEL TITLE
            st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
            image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, 1)+".bmp"
            buff1, col, buff2 = st.columns([1, 1, 1])
            # SHOW RESULTS OF IMAGES
            with col:
                showImageMarkdown(image_path)
            buff1, col, buff2 = st.columns([2.3, 2, 2])
            # SHOW DOWNLOAD
            with col:
                downloadSegmentedImage(model_name)

    # Process 1 Model
    else:
        model_name = used_model.split()[0]
        # MODEL TITLE
        st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
        image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, 1)+".bmp"
        buff1, col, buff2 = st.columns([1, 1, 1])
        # SHOW RESULTS OF IMAGES
        with col:
            showImageMarkdown(image_path)
        buff1, col, buff2 = st.columns([2.3, 2, 2])
        # SHOW DOWNLOAD
        with col:
            downloadSegmentedImage(model_name)

def showSegmentationFromFileUploader():
    # Get Used Model from Session
    used_model = st.session_state.used_model
    # Process All Models At Once
    if used_model == 'All Models At Once':
        model_names = ['UNet', 'UNet++']
        for model_name in model_names:
            # MODEL TITLE
            st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
            cols = st.columns([1, 1, 1])
            # SHOW RESULTS OF IMAGES
            for i in range(len(st.session_state.images)):
                if i == 3:
                    break
                with cols[i]:
                    image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp"
                    showImageMarkdown(image_path)
            # SHOW DOWNLOAD
            buff1, col, buff2 = st.columns([2.4, 2, 2])
            with col:
                downloadSegmentationResults(model_name)
                
    else:
        model_name = used_model.split()[0]
        # MODEL TITLE
        st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
        cols = st.columns([1, 1, 1])
        # SHOW RESULTS OF IMAGES
        for i in range(len(st.session_state.images)):
            if i == 3:
                break
            with cols[i]:
                image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp"
                showImageMarkdown(image_path)
        # SHOW DOWNLOAD
        buff1, col, buff2 = st.columns([2.4, 2, 2])
        with col:
            downloadSegmentationResults(model_name)

def zippingImage(model_name):
    folder_path = 'results_images/{}'.format(model_name)
    save_zip_path = 'results_images/results_images_{}'.format(model_name)
    shutil.make_archive(save_zip_path, 'zip', folder_path)

def downloadSegmentedImage(model_name):
    st.download_button(label='Download Segmented Image {}'.format(model_name),
            data= open('results_images/{}/segmented_images/{}_segmented_1.bmp'.format(model_name, model_name), 'rb').read(),
            file_name='results_image_{}.bmp'.format(model_name),
            mime='image/bmp')

def downloadSegmentationResults(model_name):
    zippingImage(model_name)
    st.download_button(label='Download All Images {}'.format(model_name),
            data= open('results_images/results_images_{}.zip'.format(model_name), 'rb').read(),
            file_name='results_images_{}.zip'.format(model_name),
            mime='application/zip')