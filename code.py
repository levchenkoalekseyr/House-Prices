import xgboost
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_file = "../input/house-prices-advanced-regression-techniques/train.csv"
test_file = "../input/house-prices-advanced-regression-techniques/test.csv"
train = pd.read_csv(train_file, index_col = "Id")
test = pd.read_csv(test_file, index_col = "Id")

rename_dict = {
    'MSSubClass': 'Класс_здания',
    'MSZoning': 'Зонирование',
    'LotFrontage': 'Длина_улицы',
    'LotArea': 'Площадь_участка',
    'Street': 'Тип_дороги',
    'Alley': 'Тип_съезда_с_дороги',
    'LotShape': 'Форма_участка',
    'LandContour': 'Рельеф_участка',
    'Utilities': 'Коммуникации',
    'LotConfig': 'Конфигурация_участка',
    'LandSlope': 'Уклон_участка',
    'Neighborhood': 'Район',
    'Condition1': 'Близость_к_главной_дороге1',
    'Condition2': 'Близость_к_главной_дороге2',
    'BldgType': 'Тип_жилого_дома',
    'HouseStyle': 'Стиль_дома',
    'OverallQual': 'Общее_качество_материалов',
    'OverallCond': 'Общее_состояние',
    'YearBuilt': 'Год_постройки',
    'YearRemodAdd': 'Год_ремонта',
    'RoofStyle': 'Тип_крыши',
    'RoofMatl': 'Материал_крыши',
    'Exterior1st': 'Отделка_внешняя1',
    'Exterior2nd': 'Отделка_внешняя2',
    'MasVnrType': 'Тип_облицовки_из_кирпича',
    'MasVnrArea': 'Площадь_облицовки',
    'ExterQual': 'Качество_наружней_отделки',
    'ExterCond': 'Состояние_наружней_отделки',
    'Foundation': 'Тип_фундамента',
    'BsmtQual': 'Высота_подвала',
    'BsmtCond': 'Состояние_подвала',
    'BsmtExposure': 'Окна_подвала_с_выходом_на_улицу',
    'BsmtFinType1': 'Качество_отделки_подвала_1',
    'BsmtFinSF1': 'Площадь_отделки_подвала_1',
    'BsmtFinType2': 'Качество_отделки_подвала_2',
    'BsmtFinSF2': 'Площадь_отделки_подвала_2',
    'BsmtUnfSF': 'Неотделанная_площадь_подвала',
    'TotalBsmtSF': 'Общая_площадь_подвала',
    'Heating': 'Тип_отопления',
    'HeatingQC': 'Качество_отопления',
    'CentralAir': 'Центральный_кондиционер',
    'Electrical': 'Электроснабжение',
    '1stFlrSF': 'Площадь_1_этажа',
    '2ndFlrSF': 'Площадь_2_этажа',
    'LowQualFinSF': 'Площадь_низкокачественной_отделки',
    'GrLivArea': 'Жилая_площадь_1_этажа',
    'BsmtFullBath': 'Полных_ванных_1_этаж',
    'BsmtHalfBath': 'Полуванных_1_этаж',
    'FullBath': 'Полных_ванных_выше_подвала',
    'HalfBath': 'Полуванных_выше_подвала',
    'BedroomAbvGr': 'Спальни_выше_подвала',
    'KitchenAbvGr': 'Кухни_выше_подвала',
    'KitchenQual': 'Качество_кухни',
    'TotRmsAbvGrd': 'Комнаты_выше_подвала',
    'Functional': 'Функциональность_дома',
    'Fireplaces': 'Количество_каминов',
    'FireplaceQu': 'Качество_камина',
    'GarageType': 'Тип_гаража',
    'GarageYrBlt': 'Год_постройки_гаража',
    'GarageFinish': 'Отделка_гаража',
    'GarageCars': 'Вместимость_гаража',
    'GarageArea': 'Площадь_гаража',
    'GarageQual': 'Качество_гаража',
    'GarageCond': 'Состояние_гаража',
    'PavedDrive': 'Покрытие_подъезда',
    'WoodDeckSF': 'Площадь_деревянной_террасы',
    'OpenPorchSF': 'Площадь_открытой_веранды',
    'EnclosedPorch': 'Площадь_закрытой_веранды',
    '3SsnPorch': 'Площадь_веранды_для_3_сезонов',
    'ScreenPorch': 'Площадь_экранированной_веранды',
    'PoolArea': 'Площадь_бассейна',
    'PoolQC': 'Качество_бассейна',
    'Fence': 'Тип_ограды',
    'MiscFeature': 'Дополнительная_особенность',
    'MiscVal': 'Стоимость_доп_особенности',
    'MoSold': 'Месяц_продажи',
    'YrSold': 'Год_продажи',
    'SaleType': 'Тип_продажи',
    'SaleCondition': 'Услови_продажи',
    'SalePrice': 'SalePrice'
}

train = train.rename(columns=rename_dict)
test = test.rename(columns=rename_dict)

# Только столбцы с пропусками
missing = train.isnull().sum()
missing = missing[missing > 0]
print(missing)

# тепловая карта пропусков
plt.figure(figsize=(25, 20))
sns.heatmap(train.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Тепловая карта пропусков (жёлтый = пропуск)')
plt.show()

columns_with_propusk = ['Длина_улицы', 'Тип_съезда_с_дороги', 'Тип_облицовки_из_кирпича', 'Площадь_облицовки', 'Высота_подвала', 'Состояние_подвала', 
'Окна_подвала_с_выходом_на_улицу', 'Качество_отделки_подвала_1', 'Качество_отделки_подвала_2', 'Электроснабжение', 'Качество_камина',
'Тип_гаража', 'Год_постройки_гаража', 'Отделка_гаража', 'Качество_гаража', 'Состояние_гаража', 'Качество_бассейна', 'Тип_ограды', 'Дополнительная_особенность']



train['Дополнительная_особенность'] = train['Дополнительная_особенность'].fillna('None') #Дополнительная_особенность
train.loc[train['Площадь_бассейна'].isna() | (train['Площадь_бассейна'] == '0'), 'Качество_бассейна'] = 'None' #Качество бассейна
train['Тип_ограды'] = train['Тип_ограды'].fillna('None') #Тип ограды
train['Тип_съезда_с_дороги'] = train['Тип_съезда_с_дороги'].fillna('None') #Тип съезда с дороги
train['Высота_подвала'] = train['Высота_подвала'].fillna('None') #Высота_подвала
train['Состояние_подвала'] = train['Состояние_подвала'].fillna('None') #Состояние_подвала
train['Окна_подвала_с_выходом_на_улицу'] = train['Окна_подвала_с_выходом_на_улицу'].fillna('None') #Окна_подвала_с_выходом_на_улицу
train['Качество_отделки_подвала_1'] = train['Качество_отделки_подвала_1'].fillna('None') #Качество_отделки_подвала_1
train['Качество_отделки_подвала_2'] = train['Качество_отделки_подвала_2'].fillna('None') #Качество_отделки_подвала_2

train.loc[train['Количество_каминов'].isna() | (train['Количество_каминов'] == 0), 'Качество_камина'] = 'None' #Качество камина

train['Длина_улицы'] = train['Длина_улицы'].fillna('None') #Длина улицы

train['Тип_облицовки_из_кирпича'] = train['Тип_облицовки_из_кирпича'].fillna('None') #Тип_облицовки_из_кирпича

train['Площадь_облицовки'] = train['Площадь_облицовки'].fillna(0) #Площадь_облицовки

train['Тип_гаража'] = train['Тип_гаража'].fillna('None') #Тип_гаража
train['Год_постройки_гаража'] = train['Год_постройки_гаража'].fillna('None') #Год_постройки_гаража
train['Отделка_гаража'] = train['Отделка_гаража'].fillna('None') #Отделка_гаража
train['Качество_гаража'] = train['Качество_гаража'].fillna('None') #Качество_гаража
train['Состояние_гаража'] = train['Состояние_гаража'].fillna('None') #Состояние_гаража

most_common = train['Электроснабжение'].mode()[0]  #Электроснабжение
train['Электроснабжение'] = train['Электроснабжение'].fillna(most_common)

train['Качество_бассейна'] = train['Качество_бассейна'].fillna('None') #Состояние_гаража

# тепловая карта пропусков повторно
plt.figure(figsize=(25, 20))
sns.heatmap(train.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Тепловая карта пропусков (жёлтый = пропуск)')
plt.show()

# Выбираем числовые колонки
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

# Рисуем для каждой колонки гистограмму и боксплот
for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Гистограмма с плотностью
    sns.histplot(train[col], bins=30, kde=True, ax=axes[0])
    axes[0].set_title(f'{col}')
    axes[0].set_xlabel('')
    
    # Боксплот
    sns.boxplot(x=train[col], ax=axes[1])
    axes[1].set_title(f'{col}')
    axes[1].set_xlabel('')
    
    plt.tight_layout()
    plt.show()

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

# Определяем размер сетки (3 колонки)
n_cols = 3
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))

if n_rows == 1:
    axes = axes.reshape(1, -1)

for i, col in enumerate(categorical_cols):
    row = i // n_cols
    col_idx = i % n_cols
    
    # Считаем частоты и берём топ-10 (если много категорий)
    value_counts = train[col].value_counts()
    title = col
    
    # Столбчатая диаграмма
    value_counts.plot(kind='bar', ax=axes[row, col_idx], color='coral', edgecolor='black')
    axes[row, col_idx].set_title(title, fontsize=10)
    axes[row, col_idx].set_xlabel('')
    axes[row, col_idx].set_ylabel('Количество')
    axes[row, col_idx].tick_params(axis='x', rotation=45)

# Убираем пустые графики
for i in range(len(categorical_cols), n_rows * n_cols):
    row = i // n_cols
    col_idx = i % n_cols
    fig.delaxes(axes[row, col_idx])

plt.tight_layout()
plt.show()


# матрица корреляции

# 1. Выбираем только числовые столбцы
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
train_numeric = train[numeric_cols]

# 2. Считаем корреляцию с целевой переменной
correlation_with_price = train_numeric.corr()['SalePrice'].sort_values(ascending=False)

# 3. Фильтруем признаки с корреляцией > 0.3 (по модулю)
high_corr = correlation_with_price[abs(correlation_with_price) > 0.4]

# 4. Создаём матрицу корреляции только для отобранных признаков
selected_features = high_corr.index.tolist()
corr_matrix = train_numeric[selected_features].corr()

# 5. Визуализация матрицы корреляции
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # скрываем верхнюю часть
sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    fmt='.2f', 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title('Матрица корреляции (признаки с |корреляция| > 0.3 с SalePrice)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

########################################## обработка признаков ############################################################
train['Класс_здания'] = train['Класс_здания'].astype(str) # нечисловой признак (не рейтинг), класс 2 не в 2 раза больше класса 1 и не в 2 раза меньше класса 4
test['Класс_здания'] = test['Класс_здания'].astype(str)

# 'Площадь_участка' для деревьев ничего не нужно

#'Длина_улицы' для деревьев ничего не нужно

# 'Общее_качество_материалов' ничего  не делаем

# 'Общее_состояние' ничего не делаем

# Создаем возраст дома
train['Возраст_дома'] = train['Год_продажи'] - train['Год_постройки']
test['Возраст_дома'] = test['Год_продажи'] - test['Год_постройки']

# Создаем возраст с момента последнего ремонта
train['Возраст_ремонта'] = train['Год_продажи'] - train['Год_ремонта']
test['Возраст_ремонта'] = test['Год_продажи'] - test['Год_ремонта']

# остальные числовые столбцы для деревьев ничего не требуют

##################################
print(train.columns)
#train.drop('Тип_дороги', axis=1, inplace=True) # удаляем так как почти все данные одно значение
#test.drop('Тип_дороги', axis=1, inplace=True)


#train.drop('Коммуникации', axis=1, inplace=True) # удаляем так как почти все данные одно значение
#test.drop('Коммуникации', axis=1, inplace=True)

#train.drop('Близость_к_главной_дороге2', axis=1, inplace=True) # удаляем так как почти все данные одно значение
#test.drop('Близость_к_главной_дороге2', axis=1, inplace=True)


# Примерный список редких районов, где значений < 30)
rare_neighborhoods = ['Blueste', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'IDOTRR', 
                       'MeadowV', 'Mitchel', 'NPkVill', 'NWAmes', 'OldTown', 'Sawyer', 'SawyerW', 
                       'SWISU', 'Timber', 'Veenker']

# Заменяем их на категорию 'Other_Rare'
train['Район'] = train['Район'].replace(rare_neighborhoods, 'Other_Rare')
test['Район'] = test['Район'].replace(rare_neighborhoods, 'Other_Rare')


# Конфигурация участка
train['Конфигурация_участка'] = train['Конфигурация_участка'].replace(['FR2', 'FR3'], 'Other_FR')
test['Конфигурация_участка'] = test['Конфигурация_участка'].replace(['FR2', 'FR3'], 'Other_FR')

# Уклон участка (Sev почти нет, объединяем с Mod)
train['Уклон_участка'] = train['Уклон_участка'].replace('Sev', 'Mod')
test['Уклон_участка'] = test['Уклон_участка'].replace('Sev', 'Mod')

# Близость к главной дороге 1 (выделяем редкие категории в Other)
train['Близость_к_главной_дороге1'] = train['Близость_к_главной_дороге1'].replace(['RRNn', 'PosN', 'RRAn', 'PosA'], 'Other_Cond')
test['Близость_к_главной_дороге1'] = test['Близость_к_главной_дороге1'].replace(['RRNn', 'PosN', 'RRAn', 'PosA'], 'Other_Cond')

train.drop('Материал_крыши', axis=1, inplace=True) # удаляем
test.drop('Материал_крыши', axis=1, inplace=True)

# Удаляем бесполезные столбцы
train.drop(['Тип_отопления', 'Качество_отделки_подвала_2'], axis=1, inplace=True)
test.drop(['Тип_отопления', 'Качество_отделки_подвала_2'], axis=1, inplace=True)

# 1. Тип гаража
train['Тип_гаража'] = train['Тип_гаража'].replace(['Basement', 'CarPort', '2Types'], 'Other')
test['Тип_гаража'] = test['Тип_гаража'].replace(['Basement', 'CarPort', '2Types'], 'Other')

# 2. Качество и Состояние гаража
rare_garage_qc = ['Ex', 'Gd', 'Fa', 'Po']
train['Качество_гаража'] = train['Качество_гаража'].replace(rare_garage_qc, 'Other_QC')
test['Качество_гаража'] = test['Качество_гаража'].replace(rare_garage_qc, 'Other_QC')

train['Состояние_гаража'] = train['Состояние_гаража'].replace(rare_garage_qc, 'Other_QC')
test['Состояние_гаража'] = test['Состояние_гаража'].replace(rare_garage_qc, 'Other_QC')

# Тип продажи: объединяем редкие в "Other"
rare_sale_types = ['ConLD', 'ConLI', 'ConLw', 'CWD', 'Oth', 'Con']
train['Тип_продажи'] = train['Тип_продажи'].replace(rare_sale_types, 'Other')
test['Тип_продажи'] = test['Тип_продажи'].replace(rare_sale_types, 'Other')

# Условия продажи: объединяем редкие в "Other"
rare_sale_conditions = ['Family', 'Alloca', 'AdjLand']
train['Услови_продажи'] = train['Услови_продажи'].replace(rare_sale_conditions, 'Other')
test['Услови_продажи'] = test['Услови_продажи'].replace(rare_sale_conditions, 'Other')

train = train.drop(columns=['Вместимость_гаража'])

plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], bins=50, kde=True)
plt.title('Распределение цен на жильё (SalePrice)')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()

train['SalePrice_log'] = np.log1p(train['SalePrice'])

plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice_log'], bins=50, kde=True)
plt.title('Распределение цен на жильё (SalePrice)')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()

#объединение двух DF
new_df = pd.concat([train, test], axis=0)
#print(new_df.tail)

new_df['Общая_жилая_площадь'] = new_df['Площадь_1_этажа'] + new_df['Площадь_2_этажа']
new_df['Возраст_дома'] = new_df['Год_продажи'] - new_df['Год_постройки']
new_df['Общее_количество_ванных'] = (new_df['Полных_ванных_1_этаж'] + new_df['Полных_ванных_выше_подвала'] + 
                                      0.5 * (new_df['Полуванных_1_этаж'] + new_df['Полуванных_выше_подвала']))

test['Общая_жилая_площадь'] = test['Площадь_1_этажа'] + test['Площадь_2_этажа']
test['Возраст_дома'] = test['Год_продажи'] - test['Год_постройки']
test['Общее_количество_ванных'] = (test['Полных_ванных_1_этаж'] + test['Полных_ванных_выше_подвала'] + 
                                      0.5 * (test['Полуванных_1_этаж'] + test['Полуванных_выше_подвала']))

#cat_cols = [
#    'Класс_здания', 'Зонирование', 'Тип_съезда_с_дороги', 'Форма_участка', 'Рельеф_участка',
#    'Конфигурация_участка', 'Район', 'Близость_к_главной_дороге1', 'Тип_жилого_дома',
#    'Стиль_дома', 'Тип_крыши', 'Отделка_внешняя1', 'Тип_облицовки_из_кирпича', 'Тип_фундамента',
#    'Электроснабжение', 'Центральный_кондиционер', 'Тип_гаража', 'Тип_ограды', 
#    'Дополнительная_особенность', 'Тип_продажи', 'Услови_продажи', 'Покрытие_подъезда'
#]

cat_cols = new_df.select_dtypes(include=['object']).columns

# 2. Переводим эти столбцы в категориальный тип данных Pandas
for col in cat_cols:
    # Проверяем, есть ли колонка, чтобы не упала ошибка
    if col in new_df.columns:
        new_df[col] = new_df[col].astype('category')
        test[col] = test[col].astype('category')

# Y должен браться только из данных с известными ценами!
Y_train = new_df['SalePrice_log'].iloc[:1460] # Берем только первые 1460 значений
new_df = new_df.drop('SalePrice', axis=1) 
new_df = new_df.drop('SalePrice_log', axis=1) 

# Оценку модели на валидации нужно делать через train_test_split из X_train и Y_train!
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(new_df[:1460], Y_train, test_size=0.2, random_state=42)

model = XGBRegressor(
    # Уменьшаем глубину
    max_depth=6,                 # было 8-12 → уменьшаем до 6
    min_child_weight=6,          # было 4-5 → увеличиваем
    gamma=1.0,                   # было 0.5 → увеличиваем
    
    # Увеличиваем регуляризацию
    reg_alpha=1.5,               # было 1.0 → увеличиваем
    reg_lambda=2.0,              # было 1.5 → увеличиваем
    
    # Субдискретизация
    subsample=0.6,               # было 0.7 → уменьшаем
    colsample_bytree=0.5,        # было 0.6 → уменьшаем
    
    # Скорость обучения (оставляем)
    learning_rate=0.025,
    n_estimators=2000,
    
    random_state=42,
    enable_categorical=True,
    tree_method='hist'
)

# 4. Обучаем модель
model.fit(
    X_train, Y_train,
    eval_set=[(X_val, Y_val)],
    verbose=50,
    early_stopping_rounds=50
)


# 2. Предсказываем на валидационных данных (X_val), которые мы не использовали в обучении
Y_pred_val = model.predict(X_val)

# 3. Считаем RMSE, сравнивая предсказание с реальными ценами валидации (Y_val)
rmse_val = np.sqrt(MSE(np.expm1(Y_val), np.expm1(Y_pred_val)))
print(f"Реальный RMSE на валидации: {rmse_val}")

print(set(X_val.columns) - set(test.columns))

# 1. Приводим порядок колонок в test к тому же, что и в X_train
test = test[X_train.columns]

# 2. Проверяем типы данных (они должны совпадать)
print("Типы в train:")
print(X_train.dtypes)
print("\nТипы в test:")
print(test.dtypes)

# 3. Приводим типы к одному виду
for col in X_train.columns:
    if X_train[col].dtype != test[col].dtype:
        test[col] = test[col].astype(X_train[col].dtype)


# Приводим порядок колонок в test к тому же, что в X_train
test = test[X_train.columns]

# Проверяем
print("Порядок колонок совпадает:", list(X_train.columns) == list(test.columns))

from sklearn.metrics import mean_squared_error
Y_pred_train = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(np.expm1(Y_train), np.expm1(Y_pred_train)))
rmse_val = np.sqrt(mean_squared_error(np.expm1(Y_val), np.expm1(Y_pred_val)))

print(f"RMSE трейн: ${rmse_train:,.0f}")
print(f"RMSE валидация: ${rmse_val:,.0f}")
print(f"Разница: ${rmse_val - rmse_train:,.0f}")

if rmse_val > rmse_train * 1.15:
    print("⚠️ Модель переобучена!")

Y_pred = model.predict(test)
#Y_test ['SalePrice'] = Y_pred
df = pd.DataFrame(np.expm1(Y_pred), columns=['SalePrice'])
df.index = range(1461, 2920,1)
df.index.name = "Id"
df.to_csv ('sample_submission.csv', sep=',')

