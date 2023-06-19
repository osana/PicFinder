import streamlit as st

from main import *
from setup import *

from PIL import Image
import time

def show_result(search_request, 
                search_result, 
                img_dir,
                container,
                search_time) :

    thumbnail_width = 300
    container.header("It took me "+ "{:.2f}".format(search_time)+ " sec  to find \"" +search_request+ "\" for you !")
    i = 0
    for _ in range(0, 3):
        for col in container.columns(2):
            if i >= len(search_result):
                break
            image_name, comment, score = search_result[i]

            # Загрузка изображения
            image = Image.open(img_dir + image_name)

            # Выравнивание изображения по ширине
            image_width, image_height = image.size
            aspect_ratio = thumbnail_width / image_width
            new_height = int(image_height * aspect_ratio)
            resized_image = image.resize((thumbnail_width, new_height), Image.ANTIALIAS)

            # Добавление подписи
            if score != '' :
                 sim_score = f"{float(100 * score):.2f}"
                 sim='similarity='+sim_score + "%"
                 col.markdown(comment)
                 col.markdown(f'<p style="font-size: 10px;">{sim}</p>', unsafe_allow_html=True)
            else :
                # Вывод изображения в контейнер
                col.markdown(comment)

            col.image(resized_image, width=thumbnail_width)
            i = i + 1

    return

def show_landing() :

    st.title('Find my pic!')

    search_request = st.text_input('Search for images', 
                                   'Search ...')


    col1, col2 = st.columns(2)

    if col1.button('Find!') and os.path.exists(IMAGE_DIR) :
        results = st.container()
        start_time = time.time()
        search_result = search(search_request)
        end_time = time.time()
        show_result(search_request,
                    search_result,
                    IMAGE_DIR+'/',
                    results,
                    end_time - start_time)

    if col2.button('Find with faiss!') and os.path.exists(IMAGE_DIR) :
        results = st.container()
        start_time = time.time()
        search_result = searchWithFaiss(search_request)
        end_time = time.time()
        show_result(search_request,
                    search_result,
                    IMAGE_DIR+'/',
                    results,
                    end_time - start_time)
    return


downlad_images()

show_landing()