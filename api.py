from fastapi import FastAPI
from pydantic import BaseModel
from food_detect import food_detection
from backend import generate_recipe_core

import os

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

# Инициализируем веб
app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.post("/generate-recipe")
def generate_recipe_api(request: ImageRequest):
    try:
        if TEST_MODE:
            return {"recipe": "test recipe"}

        recipe, _ = generate_recipe_core(request.image_url)

        return {
            "recipe": recipe
        }

    except Exception as e:
        return {"error": str(e)}