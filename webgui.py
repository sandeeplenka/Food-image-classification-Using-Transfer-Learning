import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model = load_model('FV.h5')
model1 = load_model(r'C:\Users\sandeep\Desktop\project\dssProject\Food-Classification-main\model.h5')


foodlabel = {0:'food',1:'non_food'}

labels = {0: 'apple', 1: 'banana', 2: 'burger', 3: 'cabbage', 4: 'carrot', 5: 'cauliflower', 6: 'cheese', 7: 'chicken_curry ', 8: 'corn', 9: 'donuts', 10: 'french_fries', 11: 'grapes', 12: 'hot_dog', 13: 'ice_cream', 14: 'mango', 15: 'momos', 16: 'omlette', 17: 'pineapple', 18: 'pizza',
          19: 'samosa', 20: 'soup', 21: 'tomato'}

healthy= ['Apple','Banana','Cabbage','Carrot','Cauliflower','Cheese','chicken_curry','Corn','Grapes','Mango','Omlette','Pineapple','Soup','Tomato']
un_healthy = ['Burger','Donuts','French_Fries','Hot_Dog','Ice_Cream','Momos','Pizza','Samosa']


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img_non_food(img_path):
    img=load_img(img_path,target_size=(256,256,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    proba = model1.predict(img)
    pred  = foodlabel[int(np.round(proba))]
    return pred


def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def run():
    st.title("Healthy And Unhealthy Food Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg","png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            is_food = processed_img_non_food(save_image_path)
            if is_food == 'food':
                result= processed_img(save_image_path)
                print(result)
                if result in healthy:
                    st.info('**Category : Healthy**')
                else:
                    st.info('**Category : Unhealthy**')
                st.success('**Predicted : '+result+'**')
                cal = fetch_calories(result)
                if cal:
                    st.warning('**'+cal+'(100 grams)**')
            else:
                st.warning('**Category : Non Food**')

run()

# py -m streamlit run webgui.py  