import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.types import BufferedInputFile
from PIL import Image
from model.model_wrapper import ModelWrapper
logging.basicConfig(level=logging.INFO)
bot = Bot(token=os.environ.get('TG_BOT_TOKEN'))
dp = Dispatcher()

user_data = {}

style_model = ModelWrapper()


class User:
    def __init__(self):
        self.style_image = None
        self.waiting_for_style = True
        self.waiting_for_content = False

    def wait_for_style(self):
        self.waiting_for_style = True
        self.waiting_for_content = False

    def wait_for_content(self):
        self.waiting_for_content = True
        self.waiting_for_style = False

    def set_style(self, image):
        self.style_image = image

    def get_style(self):
        return self.style_image


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if message.chat.id not in user_data:
        user_data[message.chat.id] = User()
    await message.answer(
        f"Привет, {message.from_user.first_name}!\n\nЭтот бот позволяет вам переносить стиль с одной картинки на другую.\n\n" +
        "Для отправки стилевого изображения используйте /select_style.\n\n" +
        "Далее можно использовать /transfer_style для стилизации изображения.\n")



@dp.message(Command("select_style"))
async def cmd_select_style(message: types.Message):
    if message.chat.id not in user_data:
        user_data[message.chat.id] = User()
    await message.answer(
        "Отправьте стилевое изображение.")
    user_data[message.chat.id].wait_for_style()


@dp.message(Command("transfer_style"))
async def cmd_transfer_style(message: types.Message):
    if message.chat.id not in user_data:
        user_data[message.chat.id] = User()
    if user_data[message.chat.id].style_image is None:
        await message.answer("Для начала необходимо отправить стилистическое изображение.")
    else:
        await message.answer(
            "Отправьте изображение, которое надо стилизовать.")
        user_data[message.chat.id].wait_for_content()


@dp.message(F.photo)
async def get_image(message: types.Message):
    if message.chat.id not in user_data:
        user_data[message.chat.id] = User()
    img = message.photo[-1]
    file_info = await bot.get_file(img.file_id)
    photo = await bot.download_file(file_info.file_path)
    image = Image.open(photo)
    if user_data[message.chat.id].waiting_for_content:
        await message.answer("Начинаю обрабатывать")
        output = style_model.process(user_data[message.chat.id].get_style(), image)
        output = BufferedInputFile(file=output.read(), filename="result.jpeg")
        await message.answer_photo(output)
    else:
        user_data[message.chat.id].set_style(image)
        user_data[message.chat.id].wait_for_content()
        await message.answer("Теперь отправьте изображение, на которое нужно перенести стиль.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
