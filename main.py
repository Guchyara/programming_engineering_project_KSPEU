# Импортирование необходимых библиотек
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import torch

from food_detect import food_detection
from recipe_generation import recipe_generator

# Загружаем изобаржение
url = "https://as2.ftcdn.net/v2/jpg/01/84/62/43/1000_F_184624337_SPmGQ04RDEDrJhbOEjO1JJxq9vfIbeot.jpg"
detected_products = food_detection(url)
print(detected_products)

# Формирование запроса к модели
prompt = f"""
You are an intelligent culinary assistant.
Generate a simple and tasty recipe using the following ingredients: {detected_products}
Write the recipe in English with clear steps and short ingredient list.
"""
recipe_generator(prompt)


