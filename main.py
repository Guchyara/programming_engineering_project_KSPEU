# Импортирование необходимых библиотек
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import torch

from food_detect import food_detection

# Загружаем изобаржение
url = "https://as2.ftcdn.net/v2/jpg/01/84/62/43/1000_F_184624337_SPmGQ04RDEDrJhbOEjO1JJxq9vfIbeot.jpg"
detected_products = food_detection(url)
print(detected_products)

