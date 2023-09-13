from fastapi import FastAPI
from pydantic import BaseModel
import fasttext.util
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()
fasttext.util.download_model('hi', if_exists='ignore')
fasttext.util.download_model('en', if_exists='ignore')


class InputData(BaseModel):
    target_string: str
    string_list: list
    language: str


@app.post("/find-closest/")
async def find_closest_string(input_data: InputData):
    target_string = input_data.target_string
    string_list = input_data.string_list
    language = input_data.language
    if language == 'hindi':
        model = fasttext.load_model('cc.en.300.bin')
    else:
        model = fasttext.load_model('cc.hi.300.bin')

    closest_string = None
    highest_similarity = -1
    target_embedding = model.get_sentence_vector(target_string)
    for string in string_list:
        string = string.lower()
        string_embedding = model.get_sentence_vector(string)
        similarity = cosine_similarity([target_embedding], [string_embedding])[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_string = string

    result = {
        'closest_string': closest_string,
        'similarity_score': highest_similarity
    }

    return result
