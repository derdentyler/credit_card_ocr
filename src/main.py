from src.utils.logger import logger
from src.data_preparation.synthetic_generator import SyntheticCardGenerator

font_paths = [
    "assets/fonts/ocra.ttf",
    "assets/fonts/CreditCard.ttf"
]

if __name__ == "__main__":
    logger.info("Генератор синтетических данных запущен")

    # Инициализация генератора
    generator = SyntheticCardGenerator(
        output_dir="datasets/synthetic_generated",  # куда сохранять изображения и метки
        font_paths=font_paths,
        background_dir="assets/backgrounds",  # папка с фоновыми изображениями
        image_size=(800, 800),  # размер выходных изображений
        card_scale=0.25,  # карта займёт 25% высоты кадра
        bg_color=(255, 255, 255),  # цвет фона, если фонов нет
        text_template="#### #### #### ####"  # шаблон номера карты
    )

    # Генерация синтетических изображений
    count = 5  # количество синтетических изображений
    generator.generate(count=count)

    logger.info("Генерация синтетических данных завершена")