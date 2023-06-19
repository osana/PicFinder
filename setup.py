
import os
import requests, zipfile, io

IMAGE_DIR = 'flickr30k_images/flickr30k_images'

def downlad_images() :

    zip_file = 'data/flickr.zip'

    #TODO : zip_file_url?
    zip_file_url = 'https://drive.google.com/open?id=14QhofCbby053kWbVeWEBHCxOROQS-bjN&authuser=0'

    try :
        if not os.path.exists(IMAGE_DIR) :
            if not os.path.exists(zip_file) :
                r = requests.get(zip_file_url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(".")

            else :
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(".")
    except :
        print("Problems with image file download")

    return