# MacroHack Yield Forecast Pipeline

Набор скриптов для перезапуска проекта с нуля: расчет факторов кривой доходности, построение прогнозных моделей и генерация итогового 6-месячного прогноза по заданию MacroHack.

## Структура проекта

- `code4.py` — полный единый скрипт пайплайна (рекомендуемый запуск).
- `main.py` — единая точка запуска.

- `data/` — исходные файлы.
- `outputs/` — итоговые графики и предсказание.

## Быстрый старт с `uv`

### Установка `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Подготовка окружения

```bash
uv init
uv sync
source .venv/bin/activate
uv pip install -r requirements.txt
```

Если `requirements.txt` ещё отсутствует, сначала установите минимальные зависимости, затем зафиксируйте их:

```bash
uv pip install numpy pandas scipy statsmodels scikit-learn matplotlib
uv pip freeze > requirements.txt
```

## Финальный сабмит в 

`outputs/Problem_1_yield_curve_predict.xlsx`