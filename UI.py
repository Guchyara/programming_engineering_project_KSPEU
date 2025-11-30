import gradio as gr
from recipe_generation import recipe_generator


def clear_history():
    return [], [], [], []

def UI_launch():
    with gr.Blocks(title="AI Culinary Assistant") as demo:
        gr.Markdown("# 🧑‍🍳 AI Кулинарный Помощник")
        gr.Markdown("Загрузите **URL** фотографии холодильника - модель определит продукты и создаст рецепт.")

        with gr.Row():
            url_input = gr.Textbox(
                label="URL изображения",
                placeholder="https://example.com/fridge.jpg",
                lines=1
            )

            generate_btn = gr.Button("Сгенерировать рецепт 🍽️")

        with gr.Row():
            recipe_out = gr.Textbox(
                label="Рецепт",
                lines=15,
                max_lines=50,
                interactive=False
            )
            img_out = gr.Image(label="Загруженное изображение")

        with gr.Accordion("📜 История ответов", open=False):
            img_history_out = gr.Gallery(
                label="История изображений", height="auto"
            )
            recipe_history_out = gr.Textbox(
                label="История рецептов",
                lines=20
            )
            clear_btn = gr.Button("Очистить историю")

        # hidden state
        img_history_state = gr.State([])
        recipe_history_state = gr.State([])

        # buttons logic
        generate_btn.click(
            recipe_generator,
            inputs=[url_input, img_history_state, recipe_history_state],
            outputs=[recipe_out, img_out, img_history_out, recipe_history_out],
        )

        clear_btn.click(
            clear_history,
            inputs=[],
            outputs=[img_history_out, recipe_history_out],
        )


    demo.launch(debug=True)