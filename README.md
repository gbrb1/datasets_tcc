# Datasets e códigos para Análise de Sentimentos

Este repositório contém os datasets e códigos utilizados para realizar a análise de sentimentos na minha tese de conclusão no curso 
Bacharelado em Sistemas de Informação na UFRRJ.

## Instruções de Uso

Para que os algoritmos de particionamento e análises sejam executados com sucesso, siga as instruções abaixo:

1. [Instale o Git](https://git-scm.com/downloads) e clone o repositório abrindo o terminal na pasta desejada e inserindo o comando

`git clone https://github.com/gbrb1/datasets_tcc.git`

2. Caso não tenha o Python instalado no seu ambiente, faça o Download da versão 3.8 ou superior
	
	[Download do Python](https://www.python.org/downloads/)

3. Ainda com o terminal aberto e com o Python já instalado, use o pip para instalar as dependências do projeto com o comando
`pip install -r requirements.txt` 

4. Faça o download do dataset sentiment140.csv em um dos links a seguir: 

	[Download pelo Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

	[Download pelo TensorFlow](https://www.tensorflow.org/datasets/catalog/sentiment140)

5. Crie uma pasta chamada `input` no mesmo nível do arquivo `particionador.py` 

6. Mova o arquivo `sentiment140.csv` para dentro da pasta e se certifique de que ele seja o primeiro a ser listado no diretório.

7. Execute o algoritmo navegando pela linha de comando até o mesmo nivel do arquivo de código python, em seguida entre com o comando

`python particionador.py` 
O algoritmo particionador utilizará o primeiro arquivo .csv encontrado dentro dessa pasta e criará um arquivo chamado `dataset_particionado.csv` 

8. Mova o arquivo `dataset_particionado.csv` para dentro da pasta `input` que se encontra no mesmo nível que o arquivo `programa_principal.py` e garanta 
que ele seja o primeiro a ser listado no diretório, se necessário alterando seu nome.

9. O algoritmo do prorama principal também irá pegar o primeiro arquivo .csv dentro da pasta `input` que se encontra no mesmo nível que ele.
Execute o algoritmo principal com o comando

`python programa_principal.py` 
