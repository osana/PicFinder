import pandas as pd
import numpy as np

def get_image_data(csv_file) :

    image_data_df = pd.read_csv (csv_file)

    image_data_df['text_embeddings'] = image_data_df['text_embeddings'].apply(lambda x: np.fromstring(x[2:-2], sep=' ')).values
    image_data_df['text_embeddings'] = image_data_df['text_embeddings'].apply(lambda x: np.reshape(x, (1, -1)))

    return image_data_df
