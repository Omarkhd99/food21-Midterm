import tensorflow as tf
import streamlit as st


def load_and_prep_image(filename, img_shape=380, scale=True):
    """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (380, 380, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 380
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.
    else:
        return img


class_names = ['apple_pie',
               'baby_back_ribs',
               'baklava',
               'bánh_bèo',
               'bánh_bột_lọc',
               'bánh_căn',
               'bánh_canh',
               'bánh_chưng',
               'bánh_cuốn',
               'bánh_đúc',
               'bánh_giò',
               'bánh_khọt',
               'bánh_mì',
               'bánh_pía',
               'bánh_tét',
               'bánh_tráng_nướng',
               'bánh_xèo',
               'beef_carpaccio',
               'beef_tartare',
               'beet_salad',
               'beignets',
               'bibimbap',
               'bread_pudding',
               'breakfast_burrito',
               'bruschetta',
               'bún_bò_huế',
               'bún_đậu_mắm_tôm',
               'bún_mắm',
               'bún_riêu',
               'bún_thịt_nướng',
               'cá_kho_tộ',
               'caesar_salad',
               'canh_chua',
               'cannoli',
               'cao_lầu',
               'caprese_salad',
               'carrot_cake',
               'ceviche',
               'cháo_lòng',
               'cheese_plate',
               'cheesecake',
               'chicken_curry',
               'chicken_quesadilla',
               'chicken_wings',
               'chocolate_cake',
               'chocolate_mousse',
               'churros',
               'clam_chowder',
               'club_sandwich',
               'cơm_tấm',
               'crab_cakes',
               'creme_brulee',
               'croque_madame',
               'cup_cakes',
               'deviled_eggs',
               'donuts',
               'dumplings',
               'edamame',
               'eggs_benedict',
               'escargots',
               'falafel',
               'filet_mignon',
               'fish_and_chips',
               'foie_gras',
               'french_fries',
               'french_onion_soup',
               'french_toast',
               'fried_calamari',
               'fried_rice',
               'frozen_yogurt',
               'garlic_bread',
               'gnocchi',
               'gỏi_cuốn',
               'greek_salad',
               'grilled_cheese_sandwich',
               'grilled_salmon',
               'guacamole',
               'gyoza',
               'hamburger',
               'hot_and_sour_soup',
               'hot_dog',
               'hủ_tiếu',
               'huevos_rancheros',
               'hummus',
               'ice_cream',
               'lasagna',
               'lobster_bisque',
               'lobster_roll_sandwich',
               'macaroni_and_cheese',
               'macarons',
               'mì_quảng',
               'miso_soup',
               'mussels',
               'nachos',
               'nem_chua',
               'omelette',
               'onion_rings',
               'oysters',
               'pad_thai',
               'paella',
               'pancakes',
               'panna_cotta',
               'peking_duck',
               'phở',
               'pizza',
               'pork_chop',
               'poutine',
               'prime_rib',
               'pulled_pork_sandwich',
               'ramen',
               'ravioli',
               'red_velvet_cake',
               'risotto',
               'samosa',
               'sashimi',
               'scallops',
               'seaweed_salad',
               'shrimp_and_grits',
               'spaghetti_bolognese',
               'spaghetti_carbonara',
               'spring_rolls',
               'steak',
               'strawberry_shortcake',
               'sushi',
               'tacos',
               'takoyaki',
               'tiramisu',
               'tuna_tartare',
               'waffles',
               'xôi_xéo']

st.markdown("<h1 style='text-align: center; color: black;'>Food Recognition App!</h1>",
            unsafe_allow_html=True)
st.write('---')
st.write('This app is capable of recognizing 130 kinds of food with over 30 kinds of Vietnamese foods '
         '(including **phở, gỏi cuốn, bánh bèo, bánh mì, hủ tiếu, bánh xèo**, etc). ')
st.write('This app is created by [Steven Truong](https://www.linkedin.com/in/luongtruong77/).')
st.write('The source codes can be found [here](https://github.com/luongtruong77/food_130_recognition)')
st.write('---')


# @st.cache(hash_funcs={tf.keras.utils.object_identity.ObjectIdentityDictionary: my_hash_func})
def model_loading(link):
    model = tf.keras.models.load_model(link)
    return model


loaded_model = model_loading(link="models/model.h5")

uploaded_file = st.file_uploader("Choose a file or use the device's camera:")

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, use_column_width=True)
    with open('./image.jpg', 'wb') as f:
        f.write(bytes_data)

    # Make predictions on custom food images

    img = load_and_prep_image("image.jpg", scale=False)  # load in target image and turn it into tensor
    pred_prob = loaded_model.predict(
        tf.expand_dims(img, axis=0))  # make prediction on image with shape [None, 224, 224, 3]
    pred_class = " ".join(class_names[pred_prob.argmax()].split("_"))  # find the predicted class label
    second_pred_prob = sorted(pred_prob[0])[-2]
    second_pred_index = list(pred_prob[0]).index(sorted(pred_prob[0])[-2])
    second_pred_class_name = " ".join(class_names[second_pred_index].split("_"))

    if pred_prob.max() <= 0.95:
        st.write(
            f"**Prediction:** {pred_prob.max() * 100:.2f}% {pred_class}, {second_pred_prob * 100:.2f}% {second_pred_class_name} ")
    else:
        st.write(
            f"**Prediction:** {pred_prob.max() * 100:.2f}% {pred_class}.")
