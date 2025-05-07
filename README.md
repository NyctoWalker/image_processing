# Image Processor (от/from 25.08.05)
## RU
**Image Processor** — приложение с открытым исходным кодом для обработки изображений на основе блочной структуры фильтров.

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
