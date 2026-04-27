from llama_cpp import Llama
from food_detect import food_detection
from PIL import Image
import requests
from io import BytesIO
import tempfile
from download_models import download_model
from backend import generate_recipe_core


# llm = Llama.from_pretrained(
#     repo_id="unsloth/gemma-3-4b-it-GGUF",
#     filename="gemma-3-4b-it-BF16.gguf",
# )


model_url = "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q2_K.gguf"
save_to = "./tensors/gemma-3-4b-it-qat-Q2_K.gguf"

download_model(model_url, save_to)


llm = Llama(
    model_path="./tensors/gemma-3-4b-it-qat-Q2_K.gguf",
    n_ctx=4096,
    # n_ctx=131072,
    # n_gpu_layers=30,
    n_gpu_layers=-1,
    verbose=False
)


# Формирование запроса к модели
def recipe_generator(url, img_history, recipe_history):
  API_URL = "http://localhost:8000/generate-recipe"

  try:
      # --- 1. запрос к backend ---
      response = requests.post(
          API_URL,
          json={"image_url": url}
      )

      data = response.json()

      if "error" in data:
          return data["error"], None, img_history, recipe_history

      recipe = "🍳 Generated recipe:\n" + data["recipe"]

      # загрузка картинки (для UI)
      img_response = requests.get(url)
      img = Image.open(BytesIO(img_response.content)).convert("RGB")

      # обновление истории
      img_history = img_history + [img]
      recipe_history = recipe_history + [recipe]

      return (
          recipe,
          img,
          img_history,
          "\n\n---\n\n".join(recipe_history),
          img_history,
          recipe_history,
      )

  except Exception as e:
      return f"❌ {str(e)}", None, img_history, "\n\n---\n\n".join(recipe_history), img_history, recipe_history
