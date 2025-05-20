# Análise de E-commerce Brasileiro Olist

## Sobre o Projeto

Este projeto implementa uma análise completa do dataset de e-commerce brasileiro da Olist, atendendo aos requisitos do desafio técnico para o Programa Trainee da triggo.ai. A solução desenvolvida realiza download automático dos dados, processamento, análise e visualização.

## Dataset

"Brazilian E-commerce Public Dataset by Olist" - Contém informações sobre 100.000 pedidos de e-commerce no Brasil realizados entre 2016 e 2018.

## Funcionalidades Implementadas

### 1. Preparação dos Dados
- Download e extração automática do dataset
- Importação para SQLite com criação de índices
- Limpeza e normalização dos dados
- Criação de modelo relacional otimizado

### 2. Análise Exploratória
- Volume de pedidos por mês e análise de sazonalidade
- Distribuição do tempo de entrega
- Relação entre frete e distância/valor
- Categorias mais vendidas por faturamento
- Estados com maior valor médio de pedido

### 3. Solução de Problemas de Negócio
- Análise de retenção de clientes (identificados 3% de clientes recorrentes)
- Predição de atraso na entrega (XGBoost com 90% de precisão)
- Segmentação de clientes (RFM) identificando 4 grupos de comportamento
- Análise detalhada da satisfação do cliente e fatores de impacto

### 4. Visualizações e Dashboards
- Dashboard geral evolutivo de vendas
- Mapa de calor de vendas por região/estado
- Análise de satisfação vs tempo de entrega
- Dashboard de análise de vendedores

## Principais Resultados

- **Retenção de Clientes**: Apenas 3% dos clientes (2.801 de 93.356) são recorrentes
- **Predição de Atrasos**: Modelo com 90% de acurácia e AUC de 0,96
- **Segmentação de Clientes**: 4 clusters distintos identificados
  - Cluster 0: 17.057 clientes (recência alta, satisfação alta)
  - Cluster 1: 17.714 clientes (recência média, satisfação baixa)
  - Cluster 2: 12.531 clientes (recência média, valor alto, satisfação alta)
  - Cluster 3: 25.571 clientes (recência baixa, satisfação máxima)
- **Satisfação vs Entrega**: Forte correlação - pedidos com entrega em 9-10 dias têm nota 5, enquanto entregas em 14+ dias têm nota 1

## Como Executar

### Opção 1: Notebook Jupyter (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/lucenfort/olistbr-brazilian-ecommerce.git
cd olistbr-brazilian-ecommerce

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook
jupyter notebook brazilian_e_commerce_triggoai.ipynb
```

### Opção 2: Script Python

```bash
# Clone o repositório
git clone https://github.com/lucenfort/olistbr-brazilian-ecommerce.git
cd olistbr-brazilian-ecommerce

# Instale as dependências
pip install -r requirements.txt

# Execute o script
python src/main.py
```

O script baixa automaticamente os dados, realiza toda a análise e gera visualizações na pasta `resultado/`.

## Estrutura do Projeto

```
olistbr-brazilian-ecommerce/
├── brazilian_e_commerce_triggoai.ipynb  # Notebook Jupyter (principal)
├── src/
│   └── main.py           # Script Python alternativo
├── data/                 # Dados baixados automaticamente
├── resultado/            # Dashboards e visualizações geradas
│   ├── dashboard_consolidado.html
│   ├── dashboard_vendas.html
│   ├── dashboard_mapa.html
│   ├── tabela_estados.html
│   └── *.png             # Visualizações estáticas
├── requirements.txt      # Dependências do projeto
└── README.md             # Este arquivo
```

## Requisitos

- Python 3.x
- Bibliotecas listadas em requirements.txt
