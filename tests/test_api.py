from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_generate_recipe_success():
    response = client.post(
        "/generate-recipe",
        json={"image_url": "https://avatars.mds.yandex.net/i?id=985eaa7823ec25b3b5d8b9d4b615757ecd86161f-5247923-images-thumbs&n=13"}
    )

    assert response.status_code == 200
    assert "recipe" in response.json()


def test_generate_recipe_invalid_url():
    response = client.post(
        "/generate-recipe",
        json={"image_url": ""}
    )

    assert response.status_code == 200
    assert "recipe" in response.json()

# Проверяем структуру ответа. Что API возвращает именно JSON ответ
def test_response_structure():
    response = client.post(
        "/generate-recipe",
        json={"image_url": "test"}
    )

    data = response.json()

    assert isinstance(data, dict)
    assert "recipe" in data
    assert isinstance(data["recipe"], str)

# Проверка отсутствия передаваемого значение в обязательном поле
def test_missing_field():
    response = client.post(
        "/generate-recipe",
        json={}
    )

    assert response.status_code == 422

# Проверка ввода неправильного типа данных (целочисленный вместо строкового)
def test_invalid_type():
    response = client.post(
        "/generate-recipe",
        json={"image_url": 123}
    )

    assert response.status_code == 422