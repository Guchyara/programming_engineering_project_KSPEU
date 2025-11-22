from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="unsloth/gemma-3-4b-it-GGUF",
    filename="gemma-3-4b-it-BF16.gguf",
)


def recipe_generator(prompt):
    output = llm(
        prompt,
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "Assistant:"]
    )

    # Вывод результата
    recipe = output["choices"][0]["text"].strip()
    print("🍳 Generated recipe:\n")
    print(recipe)
