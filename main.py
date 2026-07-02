# ==========================================================
# Исследование динамики ценовых показателей жилой недвижимости
# г. Рязань (вторичный рынок) — учебная практика
# Источник данных: выгрузка с ЦИАН (offers123.xlsx), 200 объектов
# ==========================================================
#
# ВАЖНО про источник данных:
# Читаем НАПРЯМУЮ из offers123.xlsx, а не из offers123.csv.
# CSV-версия повреждена при экспорте/конвертации:
#   - разделитель в файле фактически ';', хотя в коде "другой нейронки"
#     стояло sep=',' (как в методичке советуют считать "часто использует ','very-
#     на деле здесь не так — см. ниже)
#   - в текстовых полях (описание объявления) встречаются переносы строк,
#     из-за которых одна "логическая" строка таблицы разъезжается на
#     несколько строк CSV-файла — простым pd.read_csv это не лечится
#     надёжно без правильного quoting, и часть строк может съехать по
#     столбцам;
#   - номера телефонов в CSV превратились в числа вида 7.99E+10,
#     потому что Excel/конвертер принял длинную цифровую строку за число
#     и применил экспоненциальный формат — формат телефона необратимо
#     испорчен.
# В .xlsx эти проблемы отсутствуют: openpyxl читает ячейки как есть,
# телефоны остаются строками, переносы строк внутри ячейки не ломают
# структуру таблицы. Поэтому для всех 200 объектов используем xlsx.
# Чистить исходный xlsx вручную не нужно — все нужные преобразования
# (разбор площади, этажа, цены и т.д.) делает код ниже.

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 20)

SRC_FILE = 'offers123.xlsx'

# ==========================================================
# 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ (раздел 5 методички)
# ==========================================================

df_raw = pd.read_excel(SRC_FILE, sheet_name=0, dtype=str)
print(f"Загружено строк (объектов): {len(df_raw)}")
print(f"Столбцы исходного файла: {list(df_raw.columns)}\n")

df = pd.DataFrame()
df['id'] = df_raw['ID']
df['url'] = df_raw['Ссылка на объявление']

# --- 1.1 Цена -------------------------------------------------
# Формат: "5325000 руб., Свободная продажа, Возможна ипотека"
# Берём только цифры из первой части строки (саму сумму).
df['price'] = (
    df_raw['Цена'].astype(str)
    .str.extract(r'^(\d+)')[0]
    .astype(float)
)

# --- 1.2 Количество комнат ------------------------------------
# Формат: "2", "2, Изолированная", "3, Оба варианта, Аппартаменты" и т.п.
# Нам нужно только число комнат (1/2/3 — это и есть фильтр, который вы
# применили на ЦИАН). Заодно фиксируем отдельным флагом, является ли
# объект апартаментами — это не совсем "квартира" в юридическом смысле,
# и такие объекты обычно стоит обсудить отдельно (см. ниже).
df['rooms'] = (
    df_raw['Количество комнат'].astype(str)
    .str.extract(r'^(\d+)')[0]
    .astype(float)
)
df['is_apartments'] = df_raw['Количество комнат'].astype(str).str.contains('Аппартаменты').astype(int)

# --- 1.3 Площадь (общая/жилая/кухня) ---------------------------
# Формат поля у ЦИАН: "Общая/Жилая/Кухня", например "85.7/50.3/24.0".
# В нашей выгрузке встречаются три варианта:
#   - 2 слэша (3 числа): общая/жилая/кухня  -> 186 строк из 200
#   - 1 слэш (2 числа): общая/кухня (жилая площадь не указана продавцом)
#                                            -> 13 строк из 200
#   - 0 слэшей (1 число): только общая площадь -> 1 строка из 200
# areas[0] = общая площадь — всегда.
# areas[1] при наличии — это ЖИЛАЯ площадь, если чисел три, и КУХНЯ,
# если чисел всего два (т.к. в двухзначном варианте продавец указал
# общую и кухню, пропустив жилую). areas[2] при наличии — всегда кухня.
areas = df_raw['Площадь, м2'].astype(str).str.split('/', expand=True)
n_parts = df_raw['Площадь, м2'].astype(str).str.count('/') + 1

df['total_area'] = pd.to_numeric(areas[0], errors='coerce')

living_area = pd.Series(np.nan, index=df_raw.index, dtype=float)
kitchen_area = pd.Series(np.nan, index=df_raw.index, dtype=float)

mask_3 = n_parts == 3
living_area[mask_3] = pd.to_numeric(areas[1][mask_3], errors='coerce')
kitchen_area[mask_3] = pd.to_numeric(areas[2][mask_3], errors='coerce')

mask_2 = n_parts == 2
kitchen_area[mask_2] = pd.to_numeric(areas[1][mask_2], errors='coerce')
# living_area для mask_2 остаётся NaN — продавец её не указал

df['living_area'] = living_area
df['kitchen_area'] = kitchen_area

# --- 1.4 Этаж / этажность / материал дома -----------------------
# Формат: "2/3, Кирпичный" -> этаж/этажность, материал стен
dom_split = df_raw['Дом'].astype(str).str.extract(r'^(\d+)\s*/\s*(\d+)\s*,?\s*(.*)$')
df['floor'] = pd.to_numeric(dom_split[0], errors='coerce')
df['total_floors'] = pd.to_numeric(dom_split[1], errors='coerce')
df['material'] = dom_split[2].replace('', np.nan)

# --- 1.5 Цена за квадратный метр --------------------------------
df['price_per_sqm'] = df['price'] / df['total_area']

# --- 1.6 Фиктивные (dummy) переменные ----------------------------
df['first_floor'] = (df['floor'] == 1).astype(int)
df['last_floor'] = (df['floor'] == df['total_floors']).astype(int)
df['is_brick'] = (df['material'] == 'Кирпичный').astype(int)
df['is_panel'] = (df['material'] == 'Панельный').astype(int)
df['is_monolith'] = df['material'].isin(['Монолитный', 'Монолитно-кирпичный']).astype(int)

# Ремонт: кодируем порядковой шкалой качества (ordinal encoding)
# Без ремонта=0, Косметический=1, Евроремонт=2, Дизайнерский=3
# — сохраняем естественный порядок "лучше/хуже", что правильно для
# регрессии; альтернатива — несколько dummy, но при 4 уровнях
# и ограниченной выборке ordinal-кодирование проще и интерпретируемо.
repair_map = {'Без ремонта': 0, 'Косметический': 1, 'Евроремонт': 2, 'Дизайнерский': 3}
df['repair'] = df_raw['Ремонт'].map(repair_map)  # NaN для 6 строк без данных

# Балкон/лоджия: 1 = есть хотя бы один балкон или лоджия, 0 = нет
df['has_balcony'] = df_raw['Балкон'].notna().astype(int)

# --- 1.7 Удаление дубликатов и пропусков в базовых полях ---------
n_before = len(df)
df = df.drop_duplicates(subset='id')
n_dups = n_before - len(df)

n_before2 = len(df)
df = df.dropna(subset=['price', 'total_area', 'rooms', 'floor', 'total_floors'])
n_dropped = n_before2 - len(df)

# --- 1.8 Отсев явных выбросов / ошибок ввода ----------------------
# Этаж не может превышать этажность дома (опечатки в объявлениях
# на ЦИАН встречаются) — такие строки выбрасываем как ошибки данных.
n_before3 = len(df)
df = df[df['floor'] <= df['total_floors']]
n_floor_errors = n_before3 - len(df)

print(f"Удалено дубликатов по ID: {n_dups}")
print(f"Удалено строк с пропусками в ключевых полях: {n_dropped}")
print(f"Удалено строк с ошибкой 'этаж > этажность': {n_floor_errors}")
print(f"Итоговый размер выборки: {len(df)} объектов\n")

print("Распределение по числу комнат:")
print(df['rooms'].value_counts().sort_index())
print(f"\nИз них апартаментов: {df['is_apartments'].sum()}")
print(f"Пропусков в площади кухни: {df['kitchen_area'].isna().sum()} из {len(df)}\n")

# Логарифм площади понадобится для логарифмической модели (раздел 8)
df['log_total_area'] = np.log(df['total_area'])
df['log_price'] = np.log(df['price'])

# ==========================================================
# ПУНКТ 6. ОПИСАТЕЛЬНАЯ СТАТИСТИКА И ВИЗУАЛИЗАЦИЯ
# ==========================================================
print("=" * 60)
print("ПУНКТ 6. ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 60)

desc_cols = ['price', 'price_per_sqm', 'total_area', 'living_area', 'kitchen_area',
             'rooms', 'floor', 'total_floors', 'repair', 'has_balcony']
desc_stats = df[desc_cols].describe().round(2)
print(desc_stats)

# Мода отдельно (describe() её не считает)
print("\nМода (наиболее частое значение) по ключевым признакам:")
for c in ['rooms', 'floor', 'total_floors']:
    print(f"  {c}: {df[c].mode().iloc[0]}")

# --- Подробная интерпретация для 'price' и 'total_area' ----------
for col_name, col_label, unit in [('price', 'Стоимость квартиры', 'руб.'),
                                   ('total_area', 'Общая площадь', 'кв.м')]:
    s = df[col_name]
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    print(f"\n--- {col_label} ({unit}) ---")
    print(f"Среднее: {s.mean():.2f}")
    print(f"Медиана: {s.median():.2f}")
    print(f"Мода: {s.mode().iloc[0]:.2f}")
    print(f"Стандартное отклонение: {s.std():.2f}")
    print(f"Минимум: {s.min():.2f}, Максимум: {s.max():.2f}, Размах: {s.max()-s.min():.2f}")
    print(f"Q1={q1:.2f}, Q2(медиана)={q2:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"Границы выбросов: [{q1-1.5*iqr:.2f}; {q3+1.5*iqr:.2f}]")
    n_outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    print(f"Количество потенциальных выбросов: {n_outliers}")

# --- Гистограмма и boxplot: цена ----------------------------------
plt.figure(figsize=(10, 6))
df['price'].hist(bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Стоимость квартиры, руб.')
plt.ylabel('Частота')
plt.title('Гистограмма распределения цен на квартиры (Рязань, вторичный рынок)')
plt.tight_layout()
plt.savefig('fig_hist_price.png', dpi=120)
plt.close()

plt.figure(figsize=(8, 6))
df['price'].plot(kind='box')
plt.ylabel('Стоимость квартиры, руб.')
plt.title('Диаграмма размаха цен на квартиры')
plt.tight_layout()
plt.savefig('fig_box_price.png', dpi=120)
plt.close()

# --- Гистограмма и boxplot: общая площадь --------------------------
plt.figure(figsize=(10, 6))
df['total_area'].hist(bins=20, color='lightgreen', edgecolor='black')
plt.xlabel('Общая площадь, кв.м')
plt.ylabel('Частота')
plt.title('Гистограмма распределения общей площади квартир')
plt.tight_layout()
plt.savefig('fig_hist_area.png', dpi=120)
plt.close()

plt.figure(figsize=(8, 6))
df['total_area'].plot(kind='box')
plt.ylabel('Общая площадь, кв.м')
plt.title('Диаграмма размаха общей площади квартир')
plt.tight_layout()
plt.savefig('fig_box_area.png', dpi=120)
plt.close()

print("\nГрафики сохранены: fig_hist_price.png, fig_box_price.png, "
      "fig_hist_area.png, fig_box_area.png\n")

# ==========================================================
# ПУНКТ 7. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ==========================================================
print("=" * 60)
print("ПУНКТ 7. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 60)

# --- Диаграммы рассеяния (не менее 6 пар признаков) -----------------
scatter_pairs = [
    ('total_area', 'price', 'Общая площадь, кв.м', 'Цена, руб.'),
    ('kitchen_area', 'price', 'Площадь кухни, кв.м', 'Цена, руб.'),
    ('rooms', 'price', 'Количество комнат', 'Цена, руб.'),
    ('floor', 'price', 'Этаж', 'Цена, руб.'),
    ('total_floors', 'price', 'Этажность дома', 'Цена, руб.'),
    ('floor', 'price_per_sqm', 'Этаж', 'Цена за кв.м, руб.'),
    ('total_area', 'price_per_sqm', 'Общая площадь, кв.м', 'Цена за кв.м, руб.'),
]

for i, (x, y, xl, yl) in enumerate(scatter_pairs, start=1):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, data=df)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f'Диаграмма рассеяния: {yl} и {xl}')
    plt.tight_layout()
    plt.savefig(f'fig_scatter_{i}_{x}_vs_{y}.png', dpi=120)
    plt.close()

print(f"Построено {len(scatter_pairs)} диаграмм рассеяния (сохранены как fig_scatter_*.png)\n")

# --- Матрица корреляций и тепловая карта -----------------------------
corr_vars = ['price', 'price_per_sqm', 'total_area', 'living_area', 'kitchen_area',
             'rooms', 'floor', 'total_floors', 'repair', 'has_balcony']
corr_matrix = df[corr_vars].corr()
print("Матрица корреляций:")
print(corr_matrix.round(2))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt=".2f")
plt.title('Матрица корреляций признаков (Рязань, вторичный рынок)')
plt.tight_layout()
plt.savefig('fig_heatmap_corr.png', dpi=120)
plt.close()
print("\nТепловая карта сохранена: fig_heatmap_corr.png\n")

# --- Проверка статистической значимости корреляций с ценой -----------
print("Статистическая значимость корреляции 'price' с другими признаками "
      "(коэффициент Пирсона, t-критерий Стьюдента):\n")
sig_results = []
for var in ['total_area', 'living_area', 'kitchen_area', 'rooms', 'floor', 'total_floors',
             'repair', 'has_balcony']:
    sub = df[['price', var]].dropna()
    corr, p_value = pearsonr(sub['price'], sub[var])
    n = len(sub)
    significant = p_value < 0.05
    sig_results.append((var, corr, p_value, n, significant))
    verdict = "значима" if significant else "НЕ значима"
    print(f"  price vs {var:15s}: r={corr:+.3f}, p={p_value:.4e}, n={n}, "
          f"корреляция статистически {verdict} на уровне 5%")

print()

# ==========================================================
# ПУНКТЫ 8 И 9. РЕГРЕССИОННЫЕ МОДЕЛИ И ВЫБОР НАИЛУЧШЕЙ
# ==========================================================
print("=" * 60)
print("ПУНКТЫ 8-9. РЕГРЕССИОННЫЕ МОДЕЛИ СТОИМОСТИ ЖИЛЬЯ")
print("=" * 60)


def build_ols_model(X_columns, y_column='price', log_y=False):
    """Строит OLS-модель sm.OLS на очищенных данных по нужным столбцам."""
    needed = [y_column] + X_columns if not log_y else X_columns
    cols_for_na = list(set(X_columns + [y_column]))
    temp_df = df.dropna(subset=cols_for_na)
    y = np.log(temp_df[y_column]) if log_y else temp_df[y_column]
    X = temp_df[X_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, temp_df


models_info = {}

# Модель 1: Базовая — цена от общей площади и этажа
# Самая простая модель, проверяем, объясняет ли площадь основной разброс цен.
m1, df_m1 = build_ols_model(['total_area', 'floor'])
models_info['Модель 1 (базовая)'] = (m1, df_m1, ['total_area', 'floor'], False)

# Модель 2: Расширенная — площадь, комнаты, этажность, первый этаж
# Добавляем комнатность (влияет на планировку) и этажность дома.
m2, df_m2 = build_ols_model(['total_area', 'rooms', 'total_floors', 'first_floor'])
models_info['Модель 2 (расширенная)'] = (m2, df_m2, ['total_area', 'rooms', 'total_floors', 'first_floor'], False)

# Модель 3: + качественные характеристики жилья (ремонт, балкон, материал, этаж)
# Добавляем потребительские качества объекта — ремонт, балкон, а также
# last_floor и is_panel, чтобы явно проверить гипотезы Г3 (последний этаж)
# и Г6 (кирпич vs панель) регрессионно, а не только описательно.
m3, df_m3 = build_ols_model(['total_area', 'rooms', 'total_floors',
                              'first_floor', 'last_floor', 'is_brick', 'is_panel',
                              'repair', 'has_balcony'])
models_info['Модель 3 (+ ремонт, балкон, этаж, материал)'] = (
    m3, df_m3,
    ['total_area', 'rooms', 'total_floors', 'first_floor', 'last_floor',
     'is_brick', 'is_panel', 'repair', 'has_balcony'],
    False)

# Модель 4: Логарифмическая простая — log(цена) от log(площади) и комнат
# Логарифмирование устраняет правостороннюю асимметрию цен и площадей;
# коэффициент при log(площади) интерпретируется как эластичность.
m4, df_m4 = build_ols_model(['log_total_area', 'rooms'], log_y=True)
models_info['Модель 4 (логарифмическая)'] = (m4, df_m4, ['log_total_area', 'rooms'], True)

# Модель 5: Логарифмическая расширенная — log(цена) от log(площади),
# комнат, этажности, первого/последнего этажа, ремонта и балкона.
# Это наиболее полная спецификация — сочетает нелинейность (логарифм)
# с набором значимых предикторов.
m5, df_m5 = build_ols_model(['log_total_area', 'rooms', 'total_floors',
                              'first_floor', 'last_floor', 'repair', 'has_balcony'], log_y=True)
models_info['Модель 5 (лог. расширенная + ремонт)'] = (
    m5, df_m5,
    ['log_total_area', 'rooms', 'total_floors', 'first_floor', 'last_floor', 'repair', 'has_balcony'],
    True)

print(f"\nПостроено моделей: {len(models_info)}\n")
print(f"{'Модель':45s} {'Adj. R^2':>10s}")
print("-" * 57)
for name, (mdl, _, _, _) in models_info.items():
    print(f"{name:45s} {mdl.rsquared_adj:>10.3f}")

# --- Выбор наилучшей модели по скорректированному R^2 -----------------
best_name = max(models_info, key=lambda k: models_info[k][0].rsquared_adj)
best_model, best_df, best_Xcols, best_log_y = models_info[best_name]

print(f"\n{'=' * 60}")
print(f"НАИЛУЧШАЯ МОДЕЛЬ: {best_name}")
print(f"{'=' * 60}\n")
print(best_model.summary())

# --- Проверка значимости коэффициентов наилучшей модели -----------------
print("\nИнтерпретация значимости коэффициентов наилучшей модели:")
for factor, coef, p in zip(best_model.params.index, best_model.params, best_model.pvalues):
    status = "значим" if p < 0.05 else "НЕ значим (p > 0.05)"
    print(f"  {factor:20s}: coef={coef:>12.4f}, p={p:.4f} -> статистически {status}")

# --- Визуализация для наилучшей модели -----------------------------------
y_col = 'log_price' if best_log_y else 'price'
y_actual_model_scale = best_df[y_col] if not best_log_y else np.log(best_df['price'])
y_actual_real = best_df['price']  # фактическая цена в рублях для графика
X_best = sm.add_constant(best_df[best_Xcols])
y_pred_model_scale = best_model.predict(X_best)
y_pred_real = np.exp(y_pred_model_scale) if best_log_y else y_pred_model_scale

plt.figure(figsize=(8, 6))
plt.scatter(y_actual_real, y_pred_real, alpha=0.6, color='blue')
lims = [y_actual_real.min(), y_actual_real.max()]
plt.plot(lims, lims, 'r--')
plt.title(f'Прогнозные vs. Фактические значения ({best_name})')
plt.xlabel('Фактическая стоимость, руб.')
plt.ylabel('Прогнозная стоимость, руб.')
plt.tight_layout()
plt.savefig('fig_pred_vs_actual.png', dpi=120)
plt.close()

residuals = y_actual_real - y_pred_real
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color='orange', edgecolor='black')
plt.title(f'Гистограмма распределения остатков ({best_name})')
plt.xlabel('Ошибка (факт − прогноз), руб.')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('fig_residuals_hist.png', dpi=120)
plt.close()

print("\nГрафики наилучшей модели сохранены: fig_pred_vs_actual.png, fig_residuals_hist.png\n")

# --- Качество прогноза в "рублёвом" выражении (доп. контроль) -------------
rmse = np.sqrt(np.mean(residuals ** 2))
mae = np.mean(np.abs(residuals))
mape = np.mean(np.abs(residuals / y_actual_real)) * 100
print(f"RMSE (в рублях): {rmse:,.0f}")
print(f"MAE (в рублях):  {mae:,.0f}")
print(f"MAPE: {mape:.1f}%")

df.to_csv('ryazan_clean_dataset.csv', index=False, sep=';', encoding='utf-8-sig')
df.to_excel('ryazan_clean_dataset.xlsx', index=False)
print("\nОчищенный датасет сохранён в двух форматах:")
print("  ryazan_clean_dataset.csv  (разделитель ';')")
print("  ryazan_clean_dataset.xlsx")
print("\nГотово.")