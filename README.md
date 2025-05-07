# Image Processor (от/from 25.08.05)
## RU
**Image Processor** — приложение с открытым исходным кодом для обработки изображений на основе блочной структуры фильтров.

Для использования предлагается установить сборку приложения из списка релизов или клонировать репозиторий при помощи ```git clone https://github.com/yourusername/image-processor.git``` с установкой зависимостей (```requirements.txt```) и запуска скрипта ```main.py``` при помощи Python-компилятора.

При использовании кода решения, пожалуйста, ознакомьтесь со стандартной [лицензией](https://github.com/NyctoWalker/image_processing/blob/master/LICENSE.md).

**Основные особенности:**
- 30+ настраиваемых фильтров с поддержкой шаблонов
- Основные технологии:
  - Обработка изображений: OpenCV + NumPy
  - Интерфейс: Qt6
  - Оптимизации: Numba (JIT-компиляция для критичных участков)
- Поддерживаемые форматы: PNG, JPEG, BMP

**Ограничения:**
- Нет поддержки прозрачности
- Производительность не тестировалась системно
- Содержит как оптимизированные фильтры, так и ресурсоёмкие алгоритмы
- Для серий фильтров предложены прикладные техники оптимизации, включая кэширование в оперативной памяти (сохранение промежуточных результатов для изображения)

## EN
**Image Processor** — an open-source image processing application with modular filter architecture.

For use, it is suggested to instal build from releases list or clone repo with the following command: ```git clone https://github.com/yourusername/image-processor.git```. In that case, install libraries from ```requirements.txt``` and launch ```main.py``` with Python compilator.

If you want to use the code from here, please, check the basic [license](https://github.com/NyctoWalker/image_processing/blob/master/LICENSE.md).

**General information:**
- 30+ configurable filters with preset support
- Core technologies:
  - Image processing: OpenCV + NumPy
  - UI: Qt6 framework
  - Optimizations: Numba (JIT compilation for critical functions)
- Supported formats: PNG, JPEG, BMP

**Limitations:**
- Transparency is not supported
- No systematic performance benchmarking
- Contains both optimized and computationally intensive filters
- For series of filters there's some techniques implemented, including RAM caching with saving of intermediate image states
