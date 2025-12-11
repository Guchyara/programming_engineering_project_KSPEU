# Импортирование необходимых библиотек
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import torch
from download_models import download_model

# Загрузка модели детекции объектов
from transformers import Owlv2Processor, Owlv2ForObjectDetection

model_url = "https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/main/model.safetensors"
save_to = "./tensors/AI_For_Segmentation"

download_model(model_url, save_to)

processor = Owlv2Processor.from_pretrained("./tensors/AI_For_Segmentation")
model = Owlv2ForObjectDetection.from_pretrained("./tensors/AI_For_Segmentation")

# Функция для обнаружения еды на картинке
def food_detection(url):

  available_food_set = []
  available_food_str = ""
  common_products = [
    "milk", "eggs", "butter", "cheese", "yogurt", "cream", "sour cream", "cottage cheese",
    "chicken", "beef", "pork", "fish", "shrimp", "sausage", "bacon", "ham",
    "bread", "baguette", "tortilla", "pasta", "rice", "oats", "flour", "noodles",
    "potato", "onion", "garlic", "carrot", "tomato", "cucumber", "pepper", "cabbage",
    "banana", "coconut"
  ]

  image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

  # Предобработка изображений и детекция объектов
  inputs = processor(text=common_products, images=image, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)

  # Постобработка
  results = processor.post_process_object_detection(
    outputs=outputs,
    target_sizes=[image.size[::-1]]  # (h, w)
  )[0]

  # Вывод найденных объектов
  threshold = 0.4  # минимальная уверенность
  detected = []

  plt.figure(figsize=(8, 8))
  plt.imshow(image)
  ax = plt.gca()

  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > threshold:
        product = common_products[label]
        detected.append((product, float(score)))
        box = [round(i, 2) for i in box.tolist()]
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w-x, h-y, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f"{product} ({score:.2f})", color='lime', fontsize=10)

  plt.axis("off")
  plt.show()

  print("Найденные продукты:")
  for prod, score in detected:
    print(f"{prod} — {score:.2f}")
    available_food_set.append(prod)

  available_food_set = set(available_food_set)
  for food in available_food_set:
    available_food_str += food + " "
  return available_food_str
