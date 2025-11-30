from food_detect import food_detection
from recipe_generation import recipe_generator
import matplotlib
from UI import UI_launch
matplotlib.use("Agg")

# Хранилище истории
history_images = []
history_recipes = []

UI_launch()

