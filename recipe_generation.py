from llama_cpp import Llama
from food_detect import food_detection
from PIL import Image
import requests
from io import BytesIO
import tempfile
from download_models import download_model

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

# def recipe_generator(prompt):
#     output = llm(
#         prompt,
#         max_tokens=400,
#         temperature=0.7,
#         top_p=0.9,
#         stop=["User:", "Assistant:"]
#     )
#
#     # Вывод результата
#     recipe = output["choices"][0]["text"].strip()
#     print("🍳 Generated recipe:\n")
#     print(recipe)


# Формирование запроса к модели
def recipe_generator(url, img_history, recipe_history):
  response = requests.get(url)
  if response.status_code != 200:
    return "Ошибка загрузки изображения", None, img_history, recipe_history

  img = Image.open(BytesIO(response.content)).convert("RGB")

  # Сохраняем во временный файл, если food_detection ожидает путь
  with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    img.save("images/" +"pic_" + str(len(img_history)) + ".jpeg",format="JPEG")
    img_path = tmp.name

  food = food_detection(url)
  prompt = f"""
  You are an intelligent culinary assistant.
  Generate a simple and tasty recipe using the following ingredients: {food}
  Write the recipe in English with clear steps and short ingredient list.
  """

  # Запрос к модели
  output = llm(
    prompt,
    max_tokens=400,
    temperature=0.7,
    top_p=0.9,
    stop=["User:", "Assistant:"]
  )

  # Вывод результата
  recipe = "🍳 Generated recipe:\n" + output["choices"][0]["text"].strip()
  # Обновляем историю
  img_history = img_history + [img]
  recipe_history = recipe_history + [recipe]

  return recipe, img, img_history, recipe_history
