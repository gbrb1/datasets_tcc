import pandas as pd
import os

# Definindo as colunas e a codificação
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# Lendo o arquivo CSV
dataset_filename = os.listdir("input")[0]
dataset_path = os.path.join("input", dataset_filename)
print("Abrindo arquivo:", dataset_path)
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING)

# Pegando as linhas de 0 a 9999 e de 805000 a 814999
parte_1 = df.iloc[0:10000]
parte_2 = df.iloc[805000:815000]

# Concatenando as duas partes
df_final = pd.concat([parte_1, parte_2])

# Obtendo apenas as colunas necessárias
df_final = df_final[["target", "text"]]

# Escrevendo o resultado em um novo arquivo CSV
df_final.to_csv('dataset_particionado.csv', index=False)
