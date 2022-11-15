# pip install tensorflow-object-detection-api
from object_detection.utils import visualization_utils as viz_utils
from Test.label_map_util2 import *

import tensorflow as tf
import PIL.Image
import tensorflow_hub as hub
from PIL import Image
from six import BytesIO


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def load_image_into_numpy_array(path):
    image = None
    if(path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

@st.cache(ttl=60)
def get_model_transfert_learning():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

@st.cache(ttl=60)
def get_model_detection():
    return hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")


if true :
    pass
############# DL section #############
elif choix_page == "Deep Learning":
    # Pages
    PAGES_DL = ["Transfert de style neuronal", "Détection d'objets"]
    st.sidebar.title('Deep Learning  :brain:')
    choix_page_dl = st.sidebar.radio(label="", options=PAGES_DL)


    if choix_page_dl == "Transfert de style neuronal":
        st.markdown('<p class="grand_titre">Transfert de style neuronal</p>', unsafe_allow_html=True)
        st.write("##")
        content_path = {'Chat': 'images/tensorflow_images/chat1.jpg',
                        }
        style_path = {'La nuit étoilée - Van_Gogh': 'images/tensorflow_images/Van_Gogh1.jpg',
                      'Guernica - Picasso': 'images/tensorflow_images/GUERNICA.jpg',
                      'Le cri' : 'images/tensorflow_images/Le_cri.jpg'}
        col1, b, col2 = st.columns((1, 0.2, 1))
        with col1:
            st.markdown('<p class="section">Selectionner une image de contenu</p>', unsafe_allow_html=True)
            st.session_state.image_contenu = st.selectbox("Choisir une image", list(content_path.keys()),
                                               )
            content_image = load_img(content_path[st.session_state.image_contenu])
            content_image_plot = tf.squeeze(content_image, axis=0)
            fig = px.imshow(content_image_plot)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(
                showlegend=False,
                font=dict(size=10),
                width=600, height=300,
                margin=dict(l=40, r=50, b=40, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig)
        with col2:
            st.markdown('<p class="section">Selectionner une image de style</p>', unsafe_allow_html=True)
            st.session_state.image_style = st.selectbox("Choisir une image", list(style_path.keys()),
                                             )
            style_image = load_img(style_path[st.session_state.image_style])
            style_image_plot = tf.squeeze(style_image, axis=0)
            fig = px.imshow(style_image_plot)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(
                showlegend=False,
                font=dict(size=10),
                width=600, height=300,
                margin=dict(l=40, r=50, b=40, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig)
        if st.button("Lancer le transfert"):
            st.write("##")
            st.markdown('<p class="section">Résultat</p>', unsafe_allow_html=True)
            hub_model = get_model_transfert_learning()
            stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
            img = tensor_to_image(stylized_image)
            fig = px.imshow(img)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(
                showlegend=False,
                font=dict(size=10),
                width=1300, height=600,
                margin=dict(l=40, r=50, b=40, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig)


    elif choix_page_dl == "Détection d'objets":
        st.markdown('<p class="grand_titre">Détection d\'objets</p>', unsafe_allow_html=True)
        st.write("##")
        c1, c2 = st.columns(2)
        photo_to_detect = None
        with c1:
            choix_photo_to_detect = st.selectbox(options=["Plage", "Rue 1", "Rue 2", "Chiens", "Pont"],label="")
            if choix_photo_to_detect=="Plage":
                photo_to_detect = 'images/tensorflow_images/objects_detector/beach.jpeg'
            elif choix_photo_to_detect=="Chiens":
                photo_to_detect = 'images/tensorflow_images/objects_detector/dogs.jpeg'
            elif choix_photo_to_detect=="Rue 1":
                photo_to_detect = 'images/tensorflow_images/objects_detector/street_1.jpeg'
            elif choix_photo_to_detect=="Rue 2":
                photo_to_detect = 'images/tensorflow_images/objects_detector/street_3.png'
            elif choix_photo_to_detect=="Pont":
                photo_to_detect = 'images/tensorflow_images/objects_detector/pont_1.jpeg'
            st.write("##")
            placeholder_button = st.empty()

        st.write("##")
        placeholder_image = st.empty()
        if photo_to_detect:
            st.write("##")
            placeholder_image.image(photo_to_detect)

        if photo_to_detect :
            if placeholder_button.button("Lancer la détection"):
                image_np = load_image_into_numpy_array(photo_to_detect)
                detector = get_model_detection()
                detector_output = detector(image_np)

                COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
                                               (0, 2),
                                               (1, 3),
                                               (2, 4),
                                               (0, 5),
                                               (0, 6),
                                               (5, 7),
                                               (7, 9),
                                               (6, 8),
                                               (8, 10),
                                               (5, 6),
                                               (5, 11),
                                               (6, 12),
                                               (11, 12),
                                               (11, 13),
                                               (13, 15),
                                               (12, 14),
                                               (14, 16)]

                PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
                category_index = create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

                label_id_offset = 0
                image_np_with_detections = image_np.copy()

                result = {key: value.numpy() for key, value in detector_output.items()}

                # Use keypoints if available in detections
                keypoints, keypoint_scores = None, None
                if 'detection_keypoints' in result:
                    keypoints = result['detection_keypoints'][0]
                    keypoint_scores = result['detection_keypoint_scores'][0]

                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections[0],
                    result['detection_boxes'][0],
                    (result['detection_classes'][0] + label_id_offset).astype(int),
                    result['detection_scores'][0],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.30,
                    agnostic_mode=False,
                    keypoints=keypoints, )

                #plt.figure(figsize=(24, 32))
                im = Image.fromarray(image_np_with_detections[0])
                im.save("images/tensorflow_images/objects_detector/output.png")
                st.write("##")
                placeholder_image.image("images/tensorflow_images/objects_detector/output.png")

############# DL section #############