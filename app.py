import gradio as gr
import subprocess
import tempfile
import os
import sys


def get_python_executable():
    """Получает путь к Python исполняемому файлу с правильным окружением"""
    return sys.executable


def predict_image(image):
    try:
        # Сохраняем изображение во временный файл
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name

        # Запускаем ваш скрипт с правильным Python окружением
        result = subprocess.run(
            [get_python_executable(), 'PyTorchExecuteAI.py', temp_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # Удаляем временный файл
        os.unlink(temp_path)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            error_msg = f"Ошибка выполнения: {result.stderr}"
            print(f"Debug - stderr: {result.stderr}")
            print(f"Debug - stdout: {result.stdout}")
            return error_msg

    except Exception as e:
        return f"Ошибка при обработке: {str(e)}"


# Создание интерфейса
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Загрузите изображение цифры"),
    outputs=gr.Textbox(label="Результат распознавания",
                       lines=10,
                       max_lines=50),
    title="Распознавание цифр",
    description="Загрузите PNG изображение цифры для распознавания"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)