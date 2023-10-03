# Datasets para Análise de Sentimentos

Este repositório contém os datasets utilizados para realizar a análise de sentimentos na minha tese de conclusão de curso.

## Instruções de Uso

Para que os algoritmos de particionamento e análises sejam executados com sucesso, siga as instruções abaixo:

1. Faça o download do sentiment140.csv em um dos links a seguir: https://www.kaggle.com/datasets/kazanova/sentiment140 ou https://www.tensorflow.org/datasets/catalog/sentiment140

2. Crie uma pasta chamada "input" no mesmo nível do arquivo .py que será utilizado para o particionamento. O algoritmo utilizará o primeiro arquivo .csv encontrado dentro dessa pasta.

3. Após particionado, o algoritmo de análises também irá pegar o primeiro arquivo .csv dentro da pasta "input" que se encontra no mesmo nível que ele. 

4. Para executar os programas, navegue pela linha de comando até a pasta onde se encontra os arquivos de código-fonte .py, então entre com o comando python nome_do_arquivo.py