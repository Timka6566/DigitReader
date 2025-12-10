# bot.py (исправленный)
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

# Вставьте ваш токен
BOT_TOKEN = '8205968043:AAFolU6MqgqLxkZzK1lvA3TsQTcBEqdr0NA'
bot = telebot.TeleBot(BOT_TOKEN)

# Загрузка модели (совпадает с именем из train_model.py)
MODEL_FILENAME = 'mnist_cnn.h5'
try:
    model = load_model(MODEL_FILENAME)
    print(f"Модель {MODEL_FILENAME} загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise SystemExit(1)


def preprocess_image_for_mnist(pil_img):
    """
    Принимает PIL.Image (любой размера, цветность),
    возвращает массив shape (1,28,28,1) dtype float32, нормализованный.
    Шаги: grayscale -> автoинверсия при необходимости -> bbox crop -> resize 20x20 -> pad 28x28 -> центрирование.
    """
    # 1) серый
    img = pil_img.convert('L')

    # 2) приводим к небольшой контрастности/размерам
    # преобразуем в numpy
    arr = np.array(img)

    # 3) решаем, нужно ли инвертировать: если фон светлый (среднее > 127), значит цифра тёмная => инвертируем
    if np.mean(arr) > 127:
        arr = 255 - arr

    # 4) порог - бинаризация для поиска bbox
    # используем небольшой порог
    thresh = 30
    mask = arr > thresh
    if not np.any(mask):
        # ничего не найдено - просто ресайзим весь кадр
        img_small = Image.fromarray(arr).resize(
            (28, 28), Image.Resampling.LANCZOS)
        arr_small = np.array(img_small).astype(np.float32) / 255.0
        arr_small = np.expand_dims(arr_small, axis=(0, -1))
        return arr_small.astype(np.float32)

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 для включения границы

    crop = arr[y0:y1, x0:x1]
    pil_crop = Image.fromarray(crop)

    # 5) вписываем в 20x20 сохраняя соотношение сторон
    # вычисляем новый размер
    max_side = max(pil_crop.size)
    # scale so that largest side becomes 20
    ratio = 20.0 / max_side
    new_size = (max(1, int(pil_crop.size[0]*ratio)),
                max(1, int(pil_crop.size[1]*ratio)))
    resized = pil_crop.resize(new_size, Image.Resampling.LANCZOS)

    # 6) создаём 28x28 и вставляем в центр (фон — 0)
    new_img = Image.new('L', (28, 28), color=0)
    left = (28 - resized.size[0]) // 2
    top = (28 - resized.size[1]) // 2
    new_img.paste(resized, (left, top))

    arr_final = np.array(new_img).astype(np.float32)

    # 7) центрирование методом центра масс (улучшаем точность)
    # смещаем так, чтобы центр масс совпадал с центром изображения
    cy, cx = np.array(np.where(arr_final > 0)).mean(axis=1)
    shiftx = int(np.round(arr_final.shape[1]/2.0 - cx))
    shifty = int(np.round(arr_final.shape[0]/2.0 - cy))
    arr_final = np.roll(arr_final, shiftx, axis=1)
    arr_final = np.roll(arr_final, shifty, axis=0)

    # 8) нормализация в [0,1]
    arr_final = arr_final / 255.0

    # 9) добавляем батч и канал
    arr_final = np.expand_dims(arr_final, axis=0)   # (1,28,28)
    arr_final = np.expand_dims(arr_final, axis=-1)  # (1,28,28,1)
    return arr_final.astype(np.float32)

# Хэндлеры


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(
        message, "Привет! Отправь фото с одной цифрой (0-9) — я попробую распознать её.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
    except Exception as e:
        bot.reply_to(message, f"Ошибка загрузки фото: {e}")
        return

    try:
        img = Image.open(io.BytesIO(downloaded))
    except Exception as e:
        bot.reply_to(message, f"Невозможно открыть изображение: {e}")
        return

    try:
        x = preprocess_image_for_mnist(img)  # (1,28,28,1) float32
        preds = model.predict(x)
        predicted_digit = int(np.argmax(preds, axis=1)[0])
        confidence = float(preds[0][predicted_digit]) * 100.0

        bot.reply_to(
            message, f"Я думаю, что это цифра: *{predicted_digit}*.\nУверенность: {confidence:.2f}%", parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"Ошибка при распознавании: {e}")


@bot.message_handler(commands=['draw'])
def draw_cmd(message):
    markup = InlineKeyboardMarkup()
    btn = InlineKeyboardButton(
        "✏ Нарисовать цифру",
        web_app=WebAppInfo(url="https://timka6566.github.io/DigitReader/index.html")
    )
    markup.add(btn)
    bot.send_message(
        message.chat.id, "Открой мини-приложение:", reply_markup=markup)


@bot.message_handler(content_types=['web_app_data'])
def handle_web_app(message):
    import base64
    from PIL import Image
    import io

    data_url = message.web_app_data.data  # "data:image/png;base64,...."

    # вырезаем base64
    base64_str = data_url.split(",")[1]
    img_bytes = base64.b64decode(base64_str)

    img = Image.open(io.BytesIO(img_bytes))

    # дальше — твоя функция preprocess_image_for_mnist()
    x = preprocess_image_for_mnist(img)

    preds = model.predict(x)
    digit = int(np.argmax(preds))
    conf = float(preds[0][digit]) * 100

    bot.send_message(
        message.chat.id, f"Это цифра: *{digit}*\nУверенность: {conf:.2f}%", parse_mode="Markdown")


if __name__ == '__main__':
    print("Бот запущен.")
    bot.infinity_polling()
