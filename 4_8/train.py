import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./4_8/bug_report.csv', sep=';')
data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%Y')
data.info()
# Самостоятельно: Выведите датасет.
print(data)
# Самостоятельно: Произведите выборку по приоритету ошибок, критерий «Высокий».
print(data[data['Приоритет'] == 'Высокий'])
# Самостоятельно: Получите данные по Ошибке локализации за ноябрь 2024 года.
print(data[data['Дата'].between('2024-10-01', '2024-10-31')])
# Самостоятельно: На основе групповых вычислений для каждого типа ошибки посчитайте их количество.
print(data['Тип ошибки'].value_counts())
# Самостоятельно: Используя возможности сводной таблицы, для каждого приоритета выведите суммарное значение ошибок.
# Создаем сводную таблицу
pivot_table = pd.pivot_table(data, 
                             index='Приоритет', 
                             values='№', 
                             aggfunc='count',
                             fill_value=0)
print(pivot_table)

plt.figure(figsize=(4,3))
data['Приоритет'].value_counts().plot.bar()
plt.xlabel('Приоритет')
plt.ylabel('Количество')
plt.title('Распределение ошибок по приоритетам')
plt.show()

data['Дата'] = pd.to_datetime(data['Дата'])
errors = data.groupby(data['Дата'].dt.to_period('M'))['Дата'].count()
plt.figure(figsize=(5,3))
errors.plot()
plt.xlabel('Период')
plt.ylabel('Количество ошибок')
plt.title('Распределение ошибок по времени')
plt.show()

