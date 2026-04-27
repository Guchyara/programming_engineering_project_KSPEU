from llama_cpp import Llama
from food_detect import food_detection
from PIL import Image
import requests
from io import BytesIO

def generate_recipe_core(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Ошибка загрузки изображения")

    img = Image.open(BytesIO(response.content)).convert("RGB")

    food = food_detection(url)

    prompt = f"""
    You are an intelligent culinary assistant.
    Generate a simple and tasty recipe using the following ingredients: {food}
    """
    llm = Llama(
        model_path="./tensors/gemma-3-4b-it-qat-Q2_K.gguf",
        n_ctx=4096,
        # n_ctx=131072,
        # n_gpu_layers=30,
        n_gpu_layers=-1,
        verbose=False
    )

    output = llm(
        prompt,
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "Assistant:"]
    )

    recipe = "🍳 Generated recipe:\n" + output["choices"][0]["text"].strip()

    return recipe, img