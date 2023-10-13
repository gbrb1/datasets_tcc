# Bibliotecas Padrão
import os
import re
import numpy as np
import time

# Bibliotecas de Terceiros
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Iniciando o objeto do temporizador
process_times = {}

# Baixando pacotes do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Definindo lista de stopwords
stop_words = set(stopwords.words('english'))

# Definindo as cores utilizadas nos gráficos
cor_positivo = '#6770f5'
cor_negativo = '#ff5e77'

# Função para limpar e processar o texto


def process_text(text):
    # Limpa o texto, removendo menções a usuários,
    # URLs e caracteres não-alfanuméricos.
    text = re.sub(r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

    # Tokeniza o texto em palavras
    tokenized = word_tokenize(text)

    # Retorna uma lista de palavras, removendo aquelas que
    # estão na lista de palavras de parada (stopwords).
    return [word for word in tokenized if word.casefold() not in stop_words]


# Instanciando o analisador de sentimentos do Vader
sia = SentimentIntensityAnalyzer()

# Classificando os sentimentos usando Vader


def nltk_sentiment(text):
    text = process_text(text)
    text = ' '.join(text)
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] > 0 and sentiment['pos'] > 0:
        return "POSITIVO"
    else:
        return "NEGATIVO"

# Função para classificar os sentimentos
# usando o TextBlob


def textblob_sentiment(text):
    text = process_text(text)
    text = ' '.join(text)
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "POSITIVO"
    else:
        return "NEGATIVO"


# Definindo parâmetro para assegurar que apenas
# as colunas necessárias serão utilizadas
DATASET_COLUMNS = ["target", "text"]

# Definindo a codificação do conjunto de dados como "ISO-8859-1".
DATASET_ENCODING = "ISO-8859-1"

# Lendo o arquivo CSV
dataset_filename = os.listdir("input")[0]
dataset_path = os.path.join("input", dataset_filename)
print("Abrindo arquivo:", dataset_path)
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING,
                 names=DATASET_COLUMNS)

# Mapeando a coluna label do dataset
decode_map = {0: "NEGATIVO", 4: "POSITIVO"}

# Definindo uma função chamada "decode_sentiment" que
# recebe um rótulo como entrada e retorna um rótulo legível


def decode_sentiment(label):
    return decode_map[int(label)]


# Aplica a função "decode_sentiment" a cada elemento da coluna
# "target" do DataFrame "df".
# Isso é feito usando a função "apply" do pandas com uma função
# lambda que passa cada valor da coluna "target" para "decode_sentiment".
df.target = df.target.apply(lambda x: decode_sentiment(x))

# Definindo a proporção do conjunto de teste em relação
# ao conjunto de dados completo como 0.3 (30%).
TEST_SIZE = 0.3

# Definindo o tamanho do conjunto de treinamento como o
# complemento do tamanho do conjunto de teste (70%).
TRAIN_SIZE = 1 - TEST_SIZE

# Aplicando a função de análise de sentimentos com o Vader
# no DataFrame
start_time = time.time()
df['nltk_pred'] = df['text'].apply(nltk_sentiment)
end_time = time.time()
process_times["Vader"] = end_time - start_time

# Aplicando a função de análise de sentimentos com o TextBlob
# no DataFrame
start_time = time.time()
df['textblob_pred'] = df['text'].apply(textblob_sentiment)
end_time = time.time()
process_times["TextBlob"] = end_time - start_time

# Obtendo e imprimindo a acurácia do Vader e TextBlob
print('Acurácia do Vader:',
      accuracy_score(df['target'], df['nltk_pred']))
print('Acurácia do TextBlob:',
      accuracy_score(df['target'], df['textblob_pred']))

# Processando os textos e transformando-os em listas de strings
# que serão armazenadas na coluna processed_text
df['processed_text'] = df['text'].apply(lambda x: ' '
                                        .join(process_text(x)))

# Instanciando o vetorizador
vectorizer = TfidfVectorizer()

# Vetorizando o texto
X = vectorizer.fit_transform(df['processed_text'])
y = df['target']

# Utilizando o temporizador para obter um número inteiro
# para ser utilizado como seed no parametro random_state
current_time = int(time.time())

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=current_time)

# Definindo a grade de parâmetros para a regressão logística
param_grid = {
    'C': [1, 10, 20, 30, 40, 50, 100, 150],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [250, 500, 1000]
}

# Criando uma instância do modelo de regressão logística
start_time = time.time()
log_reg = LogisticRegression()

# Criando o objeto GridSearchCV com o modelo, os parâmetros e a validação cruzada
grid_search = GridSearchCV(
    estimator=log_reg, param_grid=param_grid, cv=5, n_jobs=-1)

# Executando a pesquisa em grade nos dados de treinamento
grid_search.fit(X_train, y_train)

# Obtendo os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_

# Usando o modelo com os melhores hiperparâmetros para fazer previsões
log_reg_best = grid_search.best_estimator_
log_reg_preds = log_reg_best.predict(X_test)
df['log_reg_pred'] = log_reg_best.predict(X)

end_time = time.time()
process_times["RL"] = end_time - start_time

param_grid = {
    'hidden_layer_sizes': [(5, 5), (10, 10), (25, 25)],
    'max_iter': [50, 100, 150],
    'alpha': [0.001],
    'learning_rate_init': [0.01],
    'solver': ['lbfgs'],
    'activation': ['relu'],
    'random_state': [current_time],
}

# Instanciando temporizador para calcular o tempo de processamento do MLP
start_time = time.time()
# Criando uma instância do modelo MLP
mlp = MLPClassifier()

# Criando um objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1)

# Realizando a busca em grade no conjunto de treinamento
grid_search.fit(X_train, y_train)

# Obtendo os melhores parâmetros encontrados pelo grid search
best_params = grid_search.best_params_

# Obtendo o melhor modelo treinado
best_mlp_model = grid_search.best_estimator_

# Fazendo previsões com o melhor modelo
mlp_preds = best_mlp_model.predict(X_test)
df['mlp_pred'] = best_mlp_model.predict(X)

end_time = time.time()
process_times["MLP"] = end_time - start_time

start_time = time.time()

param_grid = {
    'n_neighbors': [100, 151, 200],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'brute']
}

# Criando uma instância do KNeighborsClassifier
knn = KNeighborsClassifier()

# Criando uma instância do GridSearchCV
grid_search = GridSearchCV(
    estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1)

# Executando a pesquisa em grade nos dados de treinamento
grid_search.fit(X_train, y_train)

# Obtendo o melhor modelo treinado com os melhores hiperparâmetros
best_knn_model = grid_search.best_estimator_

# Obtendo os melhores hiperparâmetros encontrados pela pesquisa em grade
best_params = grid_search.best_params_

# Fazer previsões usando o melhor modelo
knn_preds = best_knn_model.predict(X_test)
df['knn_pred'] = best_knn_model.predict(X)

# Calculando o tempo de processamento
end_time = time.time()
process_times["KNN"] = end_time - start_time

# Calculando acurácia e processando relatórios das ferramentas de ML
print('Acurácia da regressão logística:',
      accuracy_score(y_test, log_reg_preds))
print('Acurácia do MLP:', accuracy_score(y_test, mlp_preds))
print('Acurácia do KNN:', accuracy_score(y_test, knn_preds))

print('Relatório de classificação da regressão logística:\n',
      classification_report(y_test, log_reg_preds, digits=4))

print('Relatório de classificação do MLP:\n',
      classification_report(y_test, mlp_preds, digits=4))

print('Relatório de classificação do KNN:\n',
      classification_report(y_test, knn_preds, digits=4))

# Calculando o F1-score e Recall para cada classe (positivo e negativo) usando o classification_report
report_nltk = classification_report(
    df['target'], df['nltk_pred'], output_dict=True)


report_textblob = classification_report(
    df['target'], df['textblob_pred'], output_dict=True)

report_log_reg = classification_report(y_test, log_reg_preds, output_dict=True)
log_reg_f1_pos = report_log_reg['POSITIVO']['f1-score']
log_reg_recall_pos = report_log_reg['POSITIVO']['recall']
log_reg_f1_neg = report_log_reg['NEGATIVO']['f1-score']
log_reg_recall_neg = report_log_reg['NEGATIVO']['recall']

report_mlp = classification_report(y_test, mlp_preds, output_dict=True)
mlp_f1_pos = report_mlp['POSITIVO']['f1-score']
mlp_recall_pos = report_mlp['POSITIVO']['recall']
mlp_f1_neg = report_mlp['NEGATIVO']['f1-score']
mlp_recall_neg = report_mlp['NEGATIVO']['recall']

report_knn = classification_report(y_test, knn_preds, output_dict=True)
knn_f1_pos = report_knn['POSITIVO']['f1-score']
knn_recall_pos = report_knn['POSITIVO']['recall']
knn_f1_neg = report_knn['NEGATIVO']['f1-score']
knn_recall_neg = report_knn['NEGATIVO']['recall']


# GRÁFICOS

# Plotando um gráfico de barras com os tempos de processamento
plt.figure(figsize=(10, 6))
times = list(process_times.values())
algorithms = list(process_times.keys())
plt.bar(algorithms, times, color=['blue', 'orange', 'green', 'purple', 'grey'])
plt.title('Tempo de Processamento por Algoritmo',
          fontsize=15, fontweight='bold')
plt.ylabel('Tempo (segundos)', fontsize=15, fontweight='bold')
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# Adicionar os valores numéricos acima das barras
for i, time in enumerate(times):
    plt.text(i, time, f'{time:.2f} s',
             ha='center', va='bottom', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.show()

# Plotando o gráfico da distribuição da polaridade dos textos
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target', data=df, order=['POSITIVO', 'NEGATIVO'], palette={
    'POSITIVO': cor_positivo, 'NEGATIVO': cor_negativo})
plt.title('Distribuição dos dados', fontsize=15,
          fontweight='bold')
plt.xlabel('Polaridade', fontsize=12, fontweight='bold')
plt.ylabel('Total de Registros', fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black', xytext=(0, 10),
                textcoords='offset points')
plt.show()

# Plotando o gráfico dos conjuntos de testes e treino

sizes = [X_train.shape[0], X_test.shape[0]]
labels = ['Treino', 'Teste']
colors = [cor_positivo, cor_negativo]
fig, ax = plt.subplots()

# Usando uma função lambda para formatar o rótulo


def format_label(p): return f'{p:.1f}%\n({int(p * sum(sizes) / 100)})'


pie = ax.pie(sizes, colors=colors, labels=labels,
             autopct=format_label, startangle=140)
ax.legend(labels, loc='upper right')
plt.title('Distribuição dos Dados de Treino e Teste',
          fontsize=15, fontweight='bold')

# Personalizando as fontes dos rótulos
for label in pie[1]:
    label.set_fontsize(12)
    label.set_fontweight('bold')

plt.show()

# Função para plotar a matriz de confusão normalizada


def plot_confusion_matrix(cm, classes, title):
    # Normalização das matrizes
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=classes,
                yticklabels=classes, cmap='Blues')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.ylabel('Verdadeiro', fontsize=12, fontweight='bold')
    plt.xlabel('Previsto', fontsize=12, fontweight='bold')
    plt.show()


# Plotando as matrizes de confusão...
nltk_cm = confusion_matrix(df['target'], df['nltk_pred'], labels=[
                           'POSITIVO', 'NEGATIVO'])
plot_confusion_matrix(
    nltk_cm, classes=['POSITIVO', 'NEGATIVO'], title='Matriz de confusão do Vader')

textblob_cm = confusion_matrix(df['target'], df['textblob_pred'], labels=[
                               'POSITIVO', 'NEGATIVO'])
plot_confusion_matrix(textblob_cm, classes=[
                      'POSITIVO', 'NEGATIVO'], title='Matriz de confusão do TextBlob')

log_reg_cm = confusion_matrix(y_test, log_reg_preds, labels=[
    'POSITIVO', 'NEGATIVO'])
plot_confusion_matrix(log_reg_cm, classes=[
                      'POSITIVO', 'NEGATIVO'], title='Matriz de confusão da Regressão Logística')

mlp_cm = confusion_matrix(y_test, mlp_preds, labels=[
    'POSITIVO', 'NEGATIVO'])
plot_confusion_matrix(mlp_cm, classes=[
                      'POSITIVO', 'NEGATIVO'], title='Matriz de confusão do MLP')


knn_cm = confusion_matrix(y_test, knn_preds, labels=[
    'POSITIVO', 'NEGATIVO'])
plot_confusion_matrix(knn_cm, classes=[
                      'POSITIVO', 'NEGATIVO'], title='Matriz de confusão do KNN')


# Plotando o gráfico de acurácias
accuracies = [accuracy_score(df['target'], df['nltk_pred']), accuracy_score(
    df['target'], df['textblob_pred']), accuracy_score(y_test, log_reg_preds), accuracy_score(y_test, mlp_preds), accuracy_score(y_test, knn_preds)]

# Multiplicando as acurácias por 100 para obter porcentagens
accuracies_percent = [accuracy * 100 for accuracy in accuracies]

plt.figure(figsize=(8, 6))

bars = plt.bar(['Vader', 'TextBlob', 'RL', 'MLP', 'KNN'],
               accuracies_percent, color=['blue', 'orange', 'green', 'purple', 'grey'])
plt.title('Acurácias', fontsize=15, fontweight='bold')
plt.ylabel('Acurácia (%)', fontsize=15, fontweight='bold')
plt.xlabel('Modelo', fontsize=15, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adicionando a acurácia em porcentagem acima das barras
for bar in bars:
    yval = round(bar.get_height(), 2)
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5,
             f'{yval}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Plotando o gráfico de barras do F1-score para cada classe
f1_scores_pos = [log_reg_f1_pos, mlp_f1_pos, knn_f1_pos]
f1_scores_neg = [log_reg_f1_neg, mlp_f1_neg, knn_f1_neg]

plt.figure(figsize=(8, 6))
bar_width = 0.35
index = np.arange(len(f1_scores_pos))

plt.bar(index, f1_scores_pos, bar_width, color=cor_positivo, label='POSITIVO')
plt.bar(index + bar_width, f1_scores_neg,
        bar_width, color=cor_negativo, label='NEGATIVO')
plt.title('F1-score por Modelo e Classe', fontsize=15, fontweight='bold')
plt.ylabel('F1-score', fontsize=15, fontweight='bold')
plt.xlabel('Modelo', fontsize=15, fontweight='bold')
plt.xticks(index + bar_width/2,
           ['Regressão Logística', 'MLP', 'KNN'], fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

# Adicionando os valores numéricos acima das barras
for i, score in enumerate(f1_scores_pos):
    plt.text(i, score, str(round(score, 2)),
             ha='center', va='bottom', fontsize=10)
for i, score in enumerate(f1_scores_neg):
    plt.text(i + bar_width, score, str(round(score, 2)),
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Plotando o gráfico de barras do Recall para cada classe
recall_scores_pos = [log_reg_recall_pos, mlp_recall_pos, knn_recall_pos]
recall_scores_neg = [log_reg_recall_neg, mlp_recall_neg, knn_recall_neg]

plt.figure(figsize=(8, 6))
bar_width = 0.35
index = np.arange(len(recall_scores_pos))

plt.bar(index, recall_scores_pos, bar_width,
        color=cor_positivo, label='POSITIVO')
plt.bar(index + bar_width, recall_scores_neg,
        bar_width, color=cor_negativo, label='NEGATIVO')
plt.title('Recall por Modelo e Classe', fontsize=15, fontweight='bold')
plt.ylabel('Recall', fontsize=15, fontweight='bold')
plt.xlabel('Modelo', fontsize=15, fontweight='bold')
plt.xticks(index + bar_width/2,
           ['Regressão Logística', 'MLP', 'KNN'], fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

# Adicionando os valores numéricos acima das barras
for i, score in enumerate(recall_scores_pos):
    plt.text(i, score, str(round(score, 2)),
             ha='center', va='bottom', fontsize=10)
for i, score in enumerate(recall_scores_neg):
    plt.text(i + bar_width, score, str(round(score, 2)),
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Gráficos de distribuição de sentimentos...

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='nltk_pred', data=df, order=['POSITIVO', 'NEGATIVO'], palette={
    'POSITIVO': cor_positivo, 'NEGATIVO': cor_negativo})
plt.title('Distribuição dos sentimentos previstos pelo Vader',
          fontsize=15, fontweight='bold')
plt.xlabel('Polaridade', fontsize=12, fontweight='bold')
plt.ylabel('Total de Predições', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black', xytext=(0, 10),
                textcoords='offset points')
plt.show()

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='textblob_pred', data=df, order=[
    'POSITIVO', 'NEGATIVO'], palette={'POSITIVO': cor_positivo, 'NEGATIVO': cor_negativo})
plt.title('Distribuição dos sentimentos previstos pelo TextBlob',
          fontsize=15, fontweight='bold')
plt.xlabel('Polaridade', fontsize=12, fontweight='bold')
plt.ylabel('Total de Predições', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black', xytext=(0, 10),
                textcoords='offset points')
plt.show()

# Filtrar o DataFrame para incluir apenas as linhas do conjunto de teste
df_test = df.iloc[y_test.index]
print(df_test['target'].value_counts())

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='log_reg_pred', data=df_test, order=['POSITIVO', 'NEGATIVO'], palette={
    'POSITIVO': cor_positivo, 'NEGATIVO': cor_negativo})
plt.title(
    'Distribuição dos sentimentos previstos pela RL no conjunto de Testes', fontsize=15, fontweight='bold')
plt.xlabel('Polaridade', fontsize=12, fontweight='bold')
plt.ylabel('Total de Predições', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black', xytext=(0, 10),
                textcoords='offset points')
plt.show()

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='mlp_pred', data=df_test, order=['POSITIVO', 'NEGATIVO'], palette={
    'POSITIVO': cor_positivo, 'NEGATIVO': cor_negativo})
plt.title(
    'Distribuição dos sentimentos previstos pelo MLP no conjunto de Testes', fontsize=15, fontweight='bold')
plt.xlabel('Polaridade', fontsize=12, fontweight='bold')
plt.ylabel('Total de Predições', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black', xytext=(0, 10),
                textcoords='offset points')
plt.show()

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='knn_pred', data=df_test, order=['POSITIVO', 'NEGATIVO'], palette={
    'POSITIVO': cor_positivo, 'NEGATIVO': cor_negativo})
plt.title(
    'Distribuição dos sentimentos previstos pelo KNN no conjunto de Testes', fontsize=15, fontweight='bold')
plt.xlabel('Polaridade', fontsize=12, fontweight='bold')
plt.ylabel('Total de Predições', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')

# Adicionando os números acima das barras
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black', xytext=(0, 10),
                textcoords='offset points')
plt.show()
