from transformers import CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from dataframe  import *

def get_model_info(model_ID, device):
    # Save the model to device
	model = CLIPModel.from_pretrained(model_ID).to(device)

    # Get the tokenizer
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    # Return model, processor & tokenizer
	return model, tokenizer


def get_single_text_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors = "pt", max_length=77, truncation=True).to(device)
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to numpy array
    embedding_as_np = text_embeddings.cpu().detach().numpy()

    return embedding_as_np

def get_item_data(result, index, measure_column) :

    img_name = str(result['image_name'][index])

    # TODO: add code to get the original comment
    comment = str(result['comment'][index])
    cos_sim = result[measure_column][index]

    return (img_name, comment, cos_sim)

def get_top_N_images(query,
                     data,
                     model, tokenizer,
                     device,
                     top_K=4) :

    query_vect = get_single_text_embedding(query, 
                                            model, tokenizer, 
                                            device)

    # Relevant columns
    relevant_cols = ["comment", "image_name", "cos_sim"]

    # Run similarity Search
    data["cos_sim"] = data["text_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))# line 17
    data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])

    data_sorted = data.sort_values(by='cos_sim', ascending=False)
    non_repeated_images = ~data_sorted["image_name"].duplicated()
    most_similar_articles = data_sorted[non_repeated_images].head(top_K)

    """
    Retrieve top_K (4 is default value) articles similar to the query
    """

    result_df = most_similar_articles[relevant_cols].reset_index()

    return [get_item_data(result_df, i, 'cos_sim') for i in range(len(result_df))]


###### with faiss ###########

import faiss
import numpy as np

def faiss_add_index_cos(df, column):

    # Get the embeddings from the specified column
    embeddings = np.vstack(df[column].values).astype(np.float32)  # Convert to float32
          
    # Create an index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings) 

    index.train(embeddings)

    # Add the embeddings to the index
    index.add(embeddings)

    # Return the index
    return index


def faiss_get_top_N_images(query,
                           data,
                           model, tokenizer,
                           device,
                           top_K=4) :

    query_vect = get_single_text_embedding(query, 
                                          model, tokenizer, 
                                          device)
    # Relevant columns
    relevant_cols = ["comment", "image_name"]

    #faiss search with cos similarity
    index = faiss_add_index_cos(data, column="text_embeddings")

    faiss.normalize_L2(query_vect)
    D, I = index.search(query_vect, len(data))

    data_sorted = data.iloc[I.flatten()]

    non_repeated_images = ~data_sorted["image_name"].duplicated()
    most_similar_articles = data_sorted[non_repeated_images].head(top_K)

    result_df = most_similar_articles[relevant_cols].reset_index()
    D = D.reshape(-1,1)[:top_K]
    result_df = pd.concat([result_df, pd.DataFrame(D, columns=['similarity'])], axis=1)
    return [get_item_data(result_df, i, 'similarity') for i in range(len(result_df))]
