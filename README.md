# Projeto de Processamento de Dados e Machine Learning

Este repositório contém um sistema modular para **processamento de dados** e **treinamento de modelos de Machine Learning**, utilizando **Python** e **scikit-learn**.  
O projeto é dividido em múltiplos experimentos (`Project1`, `Project2`, `Project3`), cada um utilizando uma base de dados diferente.

---

## 📂 Estrutura do Projeto

```plaintext
.
├── base_main.py        # Classe base principal
├── data_processing.py  # Funções de manipulação e visualização de dados
├── machine_learning.py # Funções de machine learning (modelos e pré-processamento)
├── main.py             # Arquivo principal genérico
├── project1_main.py    # Projeto 1: Previsão de Compra em Eventos
├── project2_main.py    # Projeto 2: Previsão de Conclusão de Projetos
├── project3_main.py    # Projeto 3: Previsão de Venda de Carros
└── README.md
```

---

## ⚙️ Tecnologias Utilizadas

- Python 3.10+
- Pandas
- NumPy
- scikit-learn
- Seaborn
- Matplotlib

---

## 📚 Descrição dos Projetos

### Project 1 - Previsão de Compras em Eventos (`project1_main.py`)
Treina um modelo de **SVM Linear** para prever se um participante fará uma compra baseado em seu comportamento em eventos.

- **Dataset**: `tracking.csv`
- **Features**: Inicialização, participação em palestras, contatos feitos, interesse em patrocínio.
- **Target**: Compra efetuada.

### Project 2 - Previsão de Conclusão de Projetos (`project2_main.py`)
Analisa projetos e cria modelos para prever a chance de um projeto ser finalizado.

- **Dataset**: `projetos.csv`
- **Features**: Horas esperadas e preço dos projetos.
- **Target**: Projeto finalizado.
- **Visualização**: Gráfico de dispersão (`scatterplot`).

### Project 3 - Previsão de Venda de Carros (`project3_main.py`)
Treina dois modelos (Dummy Classifier e Árvore de Decisão) para prever se um carro será vendido.

- **Dataset**: `precos.csv`
- **Features**: Preço, idade do veículo e quilometragem anual convertida para KM.
- **Target**: Carro vendido.

---

## 📈 Principais Funcionalidades

- Leitura e pré-processamento de datasets (`DataProcessing`)
- Criação de modelos de Machine Learning (`MachineLearning`)
- Treinamento e avaliação de modelos:
  - `DummyClassifier`
  - `DecisionTreeClassifier`
  - `LinearSVC`
  - `SVC`
- Visualizações com `Seaborn`
- Operações sobre os dados:
  - Normalização (`StandardScaler`)
  - Conversão de milhas para quilômetros
  - Cálculo da idade a partir do ano de fabricação
  - Filtros e mapeamentos customizados

---

## 🚀 Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/BrunoEnrico/MLSKLearnClassificationProject.git
cd MLSKLearnClassificationProject
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o projeto desejado:

```bash
python project1_main.py
```
ou

```bash
python project2_main.py
```
ou

```bash
python project3_main.py
```

---

## ✨ Melhorias Futuras

- Implementar pipelines automáticos de ML (`Pipeline` do scikit-learn).
- Adicionar métricas além de `accuracy` (Ex: `precision`, `recall`, `f1-score`).
- Salvar modelos treinados em arquivos (`.pkl`).
- Implementar cross-validation automatizado.
- Refatorar para usar logging em vez de prints.

---

## 👨‍💻 Autor

Desenvolvido por **BRUNO ENRICO**  
Contato: bruno.enrico99@gmail.com

---

## 📝 Licença

Este projeto está sob a licença [MIT](LICENSE).

