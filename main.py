import random
import torch
from dataframe import *
from model import *


def search(search_prompt : str) :

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the model ID
    model_ID = "openai/clip-vit-base-patch32"

    # Get model, processor & tokenizer
    model, tokenizer = get_model_info(model_ID, device)

    image_data_df = get_image_data('data/output2.csv')

    return get_top_N_images(search_prompt,
                            data = image_data_df,
                            model=model, tokenizer=tokenizer,
                            device = device,
                            top_K=4)

def searchWithFaiss(search_prompt : str) :

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the model ID
    model_ID = "openai/clip-vit-base-patch32"

    # Get model, processor & tokenizer
    model, tokenizer = get_model_info(model_ID, device)

    image_data_df = get_image_data('data/output2.csv')

    return faiss_get_top_N_images(search_prompt,
                                  data = image_data_df,
                                  model=model, tokenizer=tokenizer,
                                  device = device,
                                  top_K=4)