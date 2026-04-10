# MacroHack Yield Forecast Pipeline

Набор скриптов для перезапуска проекта с нуля: расчет факторов кривой доходности, построение прогнозных моделей и генерация итогового 6-месячного прогноза по заданию MacroHack.

## Структура проекта

- `code4.py` — полный единый скрипт пайплайна (рекомендуемый запуск).
- `code3.py` — резервный вариант.
- `main.py` — единая точка запуска.
- `macrohack_pipeline_demo.ipynb` — быстрый запуск и проверка результатов в Jupyter.

- `data/` — исходные файлы.
- `outputs/` — итоговые графики и таблицы диагностики.
- `figures/` — графики.

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

### Запуск пайплайна (обновлено: единый скрипт в `code4.py`)

```bash
# Новый единый запуск из code4.py
uv run python code4.py

# или через entrypoints
uv run python main.py
```

В Jupyter:

```bash
jupyter notebook macrohack_pipeline_demo.ipynb
```

## Что будет создано после запуска

- `Problem_1_yield_curve_predict.xlsx` — финальный сабмит (M1 и M2).
- `outputs/final_yc_forecasts.png` — график прогнозов кривых.
- `outputs/backtest_per_tenor.png` — график back-test.
- `outputs/tournament_rmse.png` — сводка качества всех моделей (M1/M2/VAR/Svensson/ RW).
- `outputs/macro_pca_diagnostics.png` — диагностика PCA по макро/IV факторам.
- `outputs/svensson_betas.png` — параметры Svensson betas.

## Метрики качества

- `weighted_rmse` — взвешенный RMSE по тендерам как в задании хакатона.
  - веса: `ON=0.4`, суммарно по остальным — `0.6`.
- `rmse_total = 0.5 * RMSE_M1 + 0.5 * RMSE_M2`.
  - Если есть обе версии модели (без IV и с IV), для ранжирования используется эта метрика.
  - Если доступна только одна версия — используется ее RMSE.

## Примечание по качеству экспериментов

- Проект собран как единый скрипт: легко запускать end-to-end.
- Все этапы воспроизводимы: одинаковые входные данные дают одинаковый результат.
- Гибко дорабатывается через явные блоки в `code4.py`.

## Заметки по текущим идеям (архив)

- Проверить альтернативы макро переменным и качество их заполнения (включая инфляционные ожидания).
- Продолжить сравнение VAR и ARIMA-семейств по поведению на длинных тендерах.
- Отдельно протестировать влияние Kalman-фильтра по каждой факторной схеме.
