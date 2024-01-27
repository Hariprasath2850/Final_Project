import streamlit as st
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import pandas as pd
import numpy as np
import re
import mysql.connector
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px
from streamlit_option_menu import option_menu

#python -m streamlit run Final_proj_streamlit.py

# SETTING PAGE CONFIGURATIONS
icon = Image.open("image1.jpg")
st.set_page_config(page_title="Final Project:| By HARIPRASATH ",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This web application is created to the model prediction, price prediction, Image processing and NLP *!"""})
st.markdown("<h1 style='text-align: center; color: Green;",
            unsafe_allow_html=True)

#st.snow
#python -m streamlit run Final_proj_streamlit.py


# CREATING OPTION MENU
selected = option_menu(None, ["Home", "Customer_conversion", "Product_recommendation","NLP","Image"],
                       icons=["house", "cloud-upload", "pencil-square"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "35px", "text-align": "centre", "margin": "-3px",
                                            "--hover-color": "#545454"},
                               "icon": {"font-size": "35px"},
                               "container": {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#ff5757"}})

# HOME MENU
if selected == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open("image2.png"), width=500)
        st.markdown("## :green[**Technologies Used :**] Python,easy OCR, Streamlit, SQL, Pandas")
    with col2:
        st.write(
            '## This project is the comination of Machine Learning models, NLP, Complete EDA process and Image processing ')


# Customer conversion

import streamlit as st
import pickle
import pandas as pd

data = pd.read_csv("classification_data.csv")
# Assuming you have loaded your data into 'df' dataframe

import streamlit as st
import pickle
import pandas as pd

# Load your DataFrame or define it (replace 'your_data.csv' with your actual data file)
# Example assuming you have a CSV file
df = pd.read_csv('classification_data.csv')

# Assuming you have loaded your data into 'df' dataframe

if selected == "Customer_conversion":

    # Load models
    with open('logreg_model.pkl', 'rb') as model_file:
        logreg_model = pickle.load(model_file)

    with open('knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)

    with open('xgb_model.pkl', 'rb') as model_file:
        xgb_model = pickle.load(model_file)

    with open('naive_bayes_model.pkl', 'rb') as model_file:
        naive_bayes_model = pickle.load(model_file)

    with open('decision_tree_model.pkl', 'rb') as model_file:
        decision_tree_model = pickle.load(model_file)

    with open('random_forest_model.pkl', 'rb') as model_file:
        random_forest_model = pickle.load(model_file)

    st.title("Customer Conversion Prediction App")

    # Input for additional features
    count_hit = st.number_input("Enter count_hit", min_value=0, value=0)
    channelGrouping = st.selectbox("Select Channel Grouping", df['channelGrouping'].unique())
    totals_newVisits = st.number_input("Enter totals_newVisits", min_value=0, value=0)
    
    # Additional features
    device_isMobile = st.checkbox("Is Mobile Device", value=False)
    device_deviceCategory = st.selectbox("Select Device Category", df['device_deviceCategory'].unique())
    geoNetwork_region = st.selectbox("Select GeoNetwork Region", df['geoNetwork_region'].unique())
    # Add more input fields for other features...

    # Combine user input with additional features
    user_input = [[count_hit, channelGrouping, totals_newVisits, device_isMobile, device_deviceCategory, geoNetwork_region, ...]]  # Add other features

    # Make predictions for each model
    logreg_prediction = logreg_model.predict(user_input)[0]
    knn_prediction = knn_model.predict(user_input)[0]
    xgb_prediction = xgb_model.predict(user_input)[0]
    naive_bayes_prediction = naive_bayes_model.predict(user_input)[0]
    decision_tree_prediction = decision_tree_model.predict(user_input)[0]
    random_forest_prediction = random_forest_model.predict(user_input)[0]

    # Count the number of models predicting conversion
    converted_models = sum(
        pred == 1
        for pred in [logreg_prediction, knn_prediction, xgb_prediction, naive_bayes_prediction,
                     decision_tree_prediction, random_forest_prediction]
    )

    # Decide if the customer is converted based on the majority vote
    if converted_models >= 3:
        st.write("Customer is Converted!")
    else:
        st.write("Customer is Not Converted")


#Product_Recommendation

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if selected == "Product_recommendation":

    # Create a sample dataset (replace this with your own dataset)
    data = {
        'Product': ['B001TH7GUU', 'B003ES5ZUU', 'B0019EHU8G', 'B006W8U2MU', 'B000QUUFRW'],
        'Description': ['SanDisk', 'Metal Folding Portable Laptop Stand', 'Mediabridge HDMI Cable', 'Telephone Landline Extension Cord Cable', 'Mederma Stretch Marks Therapy']
    }

    df = pd.DataFrame(data)

    # Function to get recommendations
    def get_recommendations(user_input, df):
        df['UserInput'] = user_input
        df['Combined'] = df['Description'] + ' ' + df['UserInput']
        vectorizer = CountVectorizer().fit_transform(df['Combined'])
        similarity_matrix = cosine_similarity(vectorizer, vectorizer)
        indices = pd.Series(df.index, index=df['Product']).drop_duplicates()
        idx = indices[user_input]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_recommendations = sim_scores[1:6]
        product_indices = [i[0] for i in top_recommendations]
        
        recommendations = df[['Product', 'Description']].iloc[product_indices].to_dict('records')
        return recommendations

    # Streamlit app
    def main():
        st.title("Product Recommendation")

        # User input
        user_input = st.text_input("Enter a product:", "B001TH7GUU")

        # Get recommendations on button click
        if st.button("Get Recommendations"):
            recommendations = get_recommendations(user_input, df)
            st.success("Recommended Products:")
            for rec in recommendations:
                st.write(f"Product: {rec['Product']}, Description: {rec['Description']}")

    if __name__ == "__main__":
        main()


import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import easyocr
import numpy as np
# Image processing

if selected == "Image":
    def has_text(image_path):
        # Implement your text extraction logic here using pytesseract or any other method
        pass

    def classification():
        pass

    def eda():
        pass

    def image():
        st.write("<h4 style='text-align:center; font-weight:bolder;'>Image Processing</h4>", unsafe_allow_html=True)
        upload_file = st.file_uploader('Choose a Image File', type=['png', 'jpg', 'webp'])

        if upload_file is not None:
            upload_image = np.asarray(Image.open(upload_file))
            u1 = Image.open(upload_file)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Read Original Image")
                st.image(upload_image,)
                width = st.number_input("**Enter Width**", value=(u1.size)[0])

            with col2:
                graysclae = u1.convert("L")
                st.subheader("Gray Scale Image")
                st.image(graysclae)
                height = st.number_input("**Enter Height**", value=(u1.size)[1])

        # Continue with the rest of your image processing logic...
            with col1:
                resize_image = u1.resize((int(width), int(height)))
                st.subheader("Resize Image")
                st.image(resize_image)
                radius = st.number_input("**Enter radius**", value=1)
                blur_org = u1.filter(ImageFilter.GaussianBlur(radius=int(radius)))
                st.subheader("Blurring with Original Image")
                st.image(blur_org)
                blur_gray = graysclae.filter(ImageFilter.GaussianBlur(radius=int(radius)))
                st.subheader("Blurring with Gray Scale Image")
                st.image(blur_gray)
                threshold = st.number_input("**Enter Threshold**", value=100)
                threshold_image = u1.point(lambda x: 0 if x < threshold else 255)
                st.subheader("Threshold Image")
                st.image(threshold_image)
                flip = st.selectbox("**Select Flip**", ["left-right", 'top-bottom'])
                st.subheader("Flipped Image")
                if flip == "left-right":
                    st.image(u1.transpose(Image.FLIP_LEFT_RIGHT))
                if flip == 'top-bottom':
                    st.image(u1.transpose(Image.FLIP_TOP_BOTTOM))
                brightness = st.number_input("**Enter Brightness**", value=1)
                st.subheader("Brightness Image")
                st.image((ImageEnhance.Brightness(u1)).enhance(int(brightness)))

            with col2:
                mirror_image = ImageOps.mirror(u1)
                st.subheader("Mirror Image")
                st.image(mirror_image)
                contrast = st.number_input("**Enter contrast**", value=1)
                contrast_org = ImageEnhance.Contrast(blur_org)
                st.subheader("Contrast with Original Image")
                st.image(contrast_org.enhance(int(contrast)))
                contrast_gray = ImageEnhance.Contrast(blur_gray)
                st.subheader("Contrast with Gray Scale Image")
                st.image(contrast_gray.enhance(int(contrast)))
                rotation = st.number_input("**Enter Rotation**", value=180)
                st.subheader("Rotation Image")
                st.image(u1.rotate(int(rotation)))
                sharpness = st.number_input("**Enter Sharness**", value=1)
                st.subheader("Sharpness Image")
                st.image((ImageEnhance.Sharpness(u1)).enhance(int(sharpness)))
                image_type = st.selectbox("**Select Image**", ["Original image", 'Gray Scale Image', "Blur Image",
                                                            "Threshold Image", "Sharpness Image", "Brightness Image"])

                if image_type == "Original image":
                    st.subheader("Edge Detection with Original Image")
                    st.image(u1.filter(ImageFilter.FIND_EDGES))
                if image_type == 'Gray Scale Image':
                    st.subheader("Edge Detection with Grayscale Image")
                    st.image(graysclae.filter(ImageFilter.FIND_EDGES))
                if image_type == "Blur Image":
                    st.subheader("Edge Detection with Blur Original Image")
                    st.image(blur_org.filter(ImageFilter.FIND_EDGES))

                if image_type == "Threshold Image":
                    st.subheader("Edge Detection with Threshold Image")
                    st.image(threshold_image.filter(ImageFilter.FIND_EDGES))
                if image_type == "Sharpness Image":
                    st.subheader("Edge Detection with Sharpness Image")
                    st.image(((ImageEnhance.Sharpness(u1)).enhance(int(sharpness))).filter(ImageFilter.FIND_EDGES))
                if image_type == "Brightness Image":
                    st.subheader("Edge Detection with Brightness Image")
                    st.image(((ImageEnhance.Brightness(u1)).enhance(int(brightness))).filter(ImageFilter.FIND_EDGES))

            reader = easyocr.Reader(['en'])
            bounds = reader.readtext(upload_image)
            if bounds:
                st.subheader("Extracted Text")
                file_name = upload_file.name
                if file_name == '1.png':

                    address, city = map(str, (bounds[6][1]).split(', '))
                    state, pincode = map(str, (bounds[8][1]).split())
                    image1_data = {
                        'Company': bounds[7][1] + ' ' + bounds[9][1],
                        'Card_holder_name': bounds[0][1],
                        'Desination': bounds[1][1],
                        'Mobile': bounds[2][1],
                        'Email': bounds[5][1],
                        'URL': bounds[4][1],
                        'Area': address[0:-1],
                        'City': city[0:-1],
                        'State': state,
                        'Pincode': pincode
                    }
                    st.json(image1_data)

            # Continue with other conditions...

def nlp():
    pass

def recommendation():
    pass


st.write("<h2 style='text-align:center; margin-top:-60px; font-weight:bolder;'>Final Project</h2>", unsafe_allow_html=True)

selected = st.selectbox("Select an option", ['IMAGE'])


if selected == "IMAGE":
    image()


