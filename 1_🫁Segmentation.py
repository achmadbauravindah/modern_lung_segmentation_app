import streamlit as st
import functions as f
import components.components as comp

# TITLE TAB
st.set_page_config(page_title="Segmentation 🫁")

# HEADER
def header_page():
    st.markdown("<h1 style='text-align: center;'>Modern Lung Segmentation <br> 🫁</h1>", unsafe_allow_html=True)

# FUNCTIONS
def inputFileUploaderSelected():
    st.session_state.input_selected = 'file_uploader'

def inputCameraSelected():
    st.session_state.input_selected = 'camera'

# BODY
def body_page():
    st.image('BG.png')
    # Select Model
    l_col_select_model, r_col_select_model = st.columns([2, 1])
    with l_col_select_model:
        st.markdown("<h4 style='text-align: center; margin:15px'>Select model to create lung segmentation:</h4>", unsafe_allow_html=True)
    with r_col_select_model:
        options = [
            "UNet ~ Accuracy: {}%".format(90),
            "UNet++ ~ Accuracy: {}%".format(95),
            "AttUNet ~ Accuracy: {}%".format(90),
            "DeepLabV3 ~ Accuracy: {}%".format(90),
            "All Models At Once"
        ]
        used_model = st.selectbox(label='',
                                    options=options)
        st.session_state.used_model = used_model
    # Select Input
    buff1, mid_col_input_1 , buff2, mid_col_input_2, buff3 = st.columns([2, 2, 1, 2, 2])
    btn_mid_col_input_1 = mid_col_input_1.button('Camera (Just 1 Image)', type='primary', on_click=inputCameraSelected)
    btn_mid_col_input_2 = mid_col_input_2.button('Upload File (More Images)', type='primary', on_click=inputFileUploaderSelected)
    if ('input_selected' in st.session_state):
        if ('camera' in st.session_state.input_selected):
            # Camera Input
            images = st.camera_input(label='Masukan Gambar', key='images', label_visibility='hidden')
            if images:
                f.saveSegmentation()
                f.showSegmentationFromCamera()
        
        elif ('file_uploader' in st.session_state.input_selected):
            # File Uploader Input
            images = st.file_uploader(label='Masukan Gambar', accept_multiple_files=True, key='images', type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility='hidden')
            if images:
                f.saveSegmentation()
                f.showSegmentationFromFileUploader()


# SIDEBAR
with st.sidebar:
    description = "Tujuan utama aplikasi segmentasi paru modern ini adalah untuk mengidentifikasi dan memisahkan paru-paru dari gambar radiologi, seperti Chest X-Ray (CXR). Tidak hanya itu, segmentasi paru dapat meningkatkan akurasi pada proses pelatihan model klasifikasi data citra. Citra yang diperlukan untuk pelatihan model segmentasi adalah citra Rontgen dan Mask (citra biner yang memisahkan objek dan background)."
    desc_iou = f"""
        Cara untuk menghitung keakuratan antara citra original dan hasil segmentasi adalah menggunakna Intersection Of Union (IoU). Berikut ini adalah hasil IoU beberapa model yang menggunakan 1500 data citra beserta mask-nya:
    """
    iou_all_model = f"""        
            NANTI BUAT DATAFRAME SAJA, KOLOM: MODELS DAN IOU
        """

    st.markdown(f'<p style="font-size:15px; text-align:justify">{description}', unsafe_allow_html=True)
    st.markdown("<p style='font-size:15px; text-align:justify'>{}".format(desc_iou), unsafe_allow_html=True)
    st.markdown("<span style='font-size:17px; text-align:justify font-weight:bold;'>{}</span>".format(iou_all_model), unsafe_allow_html=True)


# FOOTER
def footer_page():
    # st.markdown("<p><br></p>", unsafe_allow_html=True)
    # st.markdown("<div style='margin-top:200px;'></div>", unsafe_allow_html=True)
    comp.margin_top(200)
    st.markdown("<p style='text-align: center; font-style:italic;'>Copyright ⓒ 2023 - By Achmad Bauravindah</p>", unsafe_allow_html=True)


if __name__ == '__main__':
    header_page()
    body_page()
    footer_page()
