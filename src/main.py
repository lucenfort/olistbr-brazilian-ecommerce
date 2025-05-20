#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script em Python para análise de dados do e-commerce brasileiro Olist.

Este script implementa as seguintes funcionalidades de acordo com o desafio:

=============================================================================
DESAFIO TÉCNICO - PROGRAMA TRAINEE TRIGGO.AI
=============================================================================

1. PREPARAÇÃO DOS DADOS
   - Download e extração do dataset
   - Importação para SQLite
   - Limpeza e normalização
   - Criação de modelo relacional

2. ANÁLISE EXPLORATÓRIA DE DADOS
   - Volume de pedidos por mês e análise de sazonalidade
   - Distribuição do tempo de entrega
   - Relação frete vs distância/valor
   - Categorias mais vendidas em faturamento
   - Estados com maior valor médio de pedido

3. SOLUÇÃO DE PROBLEMAS DE NEGÓCIO
   - Análise de retenção de clientes
   - Predição de atraso na entrega
   - Segmentação de clientes (RFM)
   - Análise de satisfação do cliente

4. VISUALIZAÇÃO E DASHBOARDS
   - Dashboard geral evolutivo
   - Mapa de calor de vendas por região/estado
   - Análise de satisfação vs tempo de entrega
   - Dashboard de análise de vendedores

Autor: [Seu Nome]
Data: Maio/2025
"""
import os                              # Operações de sistema de arquivos
import zipfile                         # Compactação e descompactação de arquivos
import requests                        # Requisições HTTP
from tqdm import tqdm                  # Barra de progresso em loops
import sqlite3                         # Biblioteca para interagir com SQLite
import pandas as pd                    # Biblioteca para manipulação de dados
import numpy as np                     # Biblioteca para operações numéricas
import matplotlib.pyplot as plt        # Biblioteca para geração de gráficos
import seaborn as sns                  # Biblioteca para visualização estatística
import plotly.express as px            # Biblioteca para dashboards interativos
from plotly.subplots import make_subplots  # Para compor múltiplos gráficos
from sklearn.model_selection import train_test_split  # Divisão treino/teste
from sklearn.metrics import classification_report, roc_auc_score, roc_curve  # Métricas
from sklearn.cluster import KMeans     # Algoritmo de clustering
from sklearn.preprocessing import StandardScaler  # Normalização de features
from imblearn.over_sampling import SMOTE  # Para balanceamento das classes
from sklearn.metrics import silhouette_score  # Para avaliar clusters
import plotly.graph_objects as go      # Para gráficos interativos
import time                           # Para medição de tempo
from datetime import datetime         # Para formatação de datas
import warnings                       # Para suprimir warnings
import xgboost as xgb                 # Para modelo de classificação
import shutil                        # Para manipulação de diretórios

# Suprimir warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS 
# =============================================================================
# Esta seção inclui a importação, limpeza e transformação dos dados originais.
# Inclui a criação das tabelas, conexões, índices e features derivadas.
# =============================================================================

# --------------------------------------------------------
# Configurações iniciais
# --------------------------------------------------------
BASE_PATH = "./data"                      # Diretório base onde estão os arquivos CSV
DB_PATH   = "olist.db"                    # Caminho do arquivo de banco SQLite
RESULTS_PATH = "resultado"                # Diretório para salvar os gráficos

# Função para limpar a pasta de resultados
def clean_results_folder():
    """Limpa a pasta de resultados para garantir dados atualizados."""
    if os.path.exists(RESULTS_PATH):
        try:
            shutil.rmtree(RESULTS_PATH)
            print(f"Pasta {RESULTS_PATH} removida com sucesso.")
        except Exception as e:
            print(f"Erro ao limpar pasta de resultados: {e}")
    
    # Cria diretório de resultados
    os.makedirs(RESULTS_PATH, exist_ok=True)
    print(f"Pasta {RESULTS_PATH} criada com sucesso.")

# Limpar pasta de resultados antes de iniciar
clean_results_folder()

# Configuração para salvar arquivos
def save_file(fig, filename):
    """Função segura para salvar arquivos."""
    try:
        full_path = os.path.join(RESULTS_PATH, filename)
        fig.savefig(full_path)
        plt.close(fig)
        print(f"Arquivo salvo: {full_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar {filename}: {e}")
        plt.close(fig)
        return False

# Configuração de cores padrão para todos os gráficos
COLORS = {
    'primary': '#1f77b4',    # Azul
    'secondary': '#2ca02c',  # Verde
    'tertiary': '#ff7f0e',   # Laranja
    'quaternary': '#d62728', # Vermelho
    'background': '#f8f9fa', # Cinza claro
    'grid': '#e9ecef',       # Cinza mais claro
    'text': '#2c3e50'        # Azul escuro
}

# Configuração de estilo para os gráficos
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': COLORS['grid'],
    'axes.facecolor': COLORS['background'],
    'axes.edgecolor': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

# Configuração de cores para seaborn
sns.set_palette("husl")
sns.set_style("whitegrid")

# Função para medir tempo de execução
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"\nIniciando {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tempo de execução de {func.__name__}: {execution_time:.2f} segundos")
        
        # Salva o tempo de execução
        with open(os.path.join(RESULTS_PATH, 'execution_times.txt'), 'a') as f:
            f.write(f"{datetime.now()}: {func.__name__} - {execution_time:.2f} segundos\n")
        
        return result
    return wrapper

@timer_decorator
def download_and_extract_dataset():
    """
    Função para baixar e extrair o dataset do Kaggle.
    Retorna o tempo de execução para métricas.
    """
    # Definir diretório de destino
    data_dir = 'data'
    zip_path = os.path.join(data_dir, 'brazilian-ecommerce.zip')
    url = 'https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce'

    # Criar diretório se não existir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        # Baixar o dataset com barra de progresso
        print('Baixando o dataset...')
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
        progress_bar.close()
        print('Download concluído.')

        # Descompactar o arquivo
        print('Descompactando o arquivo...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print('Descompactação concluída.')

        # Fechar explicitamente o arquivo ZIP antes de tentar removê-lo
        zip_ref.close()
        
        # Aguardar um momento para garantir que o arquivo foi liberado
        time.sleep(1)
        
        # Tentar remover o arquivo ZIP
        try:
            os.remove(zip_path)
            print('Arquivo ZIP removido.')
        except PermissionError:
            print('Aviso: Não foi possível remover o arquivo ZIP. Ele pode estar em uso.')
        except Exception as e:
            print(f'Aviso: Erro ao tentar remover o arquivo ZIP: {str(e)}')

        print('Dataset da Olist baixado e organizado com sucesso na pasta data/.')
        
    except Exception as e:
        print(f'Erro durante o download ou extração: {str(e)}')
        raise

# Executa o download e extração
download_and_extract_dataset()

# --------------------------------------------------------
# Função para importar CSVs para tabelas SQLite
# (Tarefa 1: Preparação dos Dados - Importação dos dados)
# --------------------------------------------------------
@timer_decorator
def import_csv_to_sqlite(csv_path, table_name, conn):
    """
    Lê um CSV e armazena seu conteúdo em uma tabela SQLite.
    Corrige nomes de colunas específicas antes de gravar.
    Se a tabela já existir, ela será substituída.
    
    Args:
        csv_path (str): Caminho do arquivo CSV
        table_name (str): Nome da tabela no SQLite
        conn (sqlite3.Connection): Conexão com o banco de dados
    """
    # Lê CSV usando pandas
    df = pd.read_csv(csv_path)

    # Corrige nome de coluna com erro de digitação no arquivo de produtos
    if table_name == 'products' and 'product_name_lenght' in df.columns:
        df.rename(columns={'product_name_lenght': 'product_name_length'}, inplace=True)

    # Grava DataFrame em tabela SQLite, substituindo se já existir
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Tabela '{table_name}' importada: {len(df)} registros.")

@timer_decorator
def setup_database():
    """
    Configura o banco de dados SQLite e importa todos os datasets.
    Cria índices e views necessárias para otimização.
    """
    # Conecta (ou cria) o banco de dados SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Importação dos datasets principais
    datasets = [
        ("olist_order_items_dataset.csv", "order_items"),
        ("olist_order_reviews_dataset.csv", "order_reviews"),
        ("olist_orders_dataset.csv", "orders"),
        ("olist_products_dataset.csv", "products"),
        ("olist_geolocation_dataset.csv", "geolocation"),
        ("olist_sellers_dataset.csv", "sellers"),
        ("olist_order_payments_dataset.csv", "payments"),
        ("olist_customers_dataset.csv", "customers"),
        ("product_category_name_translation.csv", "category_translation")
    ]

    for csv_file, table_name in datasets:
        import_csv_to_sqlite(f"{BASE_PATH}/{csv_file}", table_name, conn)

    # Criação de índices para otimização
    cursor.executescript("""
    CREATE INDEX IF NOT EXISTS idx_orders_order_id       ON orders(order_id);
    CREATE INDEX IF NOT EXISTS idx_items_order_id        ON order_items(order_id);
    CREATE INDEX IF NOT EXISTS idx_payments_order_id     ON payments(order_id);
    CREATE INDEX IF NOT EXISTS idx_reviews_order_id      ON order_reviews(order_id);
    CREATE INDEX IF NOT EXISTS idx_products_product_id   ON products(product_id);
    CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);
    CREATE INDEX IF NOT EXISTS idx_sellers_seller_id     ON sellers(seller_id);
    """)
    conn.commit()

    # Criação da view para dados mesclados
    cursor.execute("DROP VIEW IF EXISTS merged_data")
    cursor.execute("""
    CREATE VIEW merged_data AS
    SELECT
        o.order_id,
        o.customer_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        i.order_item_id,
        i.product_id,
        i.seller_id,
        i.shipping_limit_date,
        i.price,
        i.freight_value,
        pay.payment_sequential,
        pay.payment_type,
        pay.payment_installments,
        pay.payment_value,
        r.review_id,
        r.review_score,
        r.review_comment_title,
        r.review_comment_message,
        r.review_creation_date,
        r.review_answer_timestamp,
        p.product_category_name,
        p.product_name_length,
        p.product_description_lenght,
        p.product_photos_qty,
        p.product_weight_g,
        p.product_length_cm,
        p.product_height_cm,
        p.product_width_cm,
        c.customer_unique_id,
        c.customer_zip_code_prefix,
        c.customer_city,
        c.customer_state,
        s.seller_zip_code_prefix,
        s.seller_city,
        s.seller_state
    FROM orders o
    INNER JOIN order_items i           ON o.order_id = i.order_id
    INNER JOIN payments pay            ON o.order_id = pay.order_id
    INNER JOIN order_reviews r         ON o.order_id = r.order_id
    INNER JOIN products p              ON i.product_id = p.product_id
    INNER JOIN customers c             ON o.customer_id = c.customer_id
    INNER JOIN sellers s               ON i.seller_id = s.seller_id
    """)
    conn.commit()

    return conn

# Executa a configuração do banco de dados
conn = setup_database()

# --------------------------------------------------------
# Leitura da view mesclada para um DataFrame pandas
# --------------------------------------------------------
df = pd.read_sql_query("SELECT * FROM merged_data", conn)

# Verifica as colunas disponíveis
print("\nColunas disponíveis no DataFrame:")
print(df.columns.tolist())

# --------------------------------------------------------
# Limpeza e preparação dos dados
# (Tarefa 1: Preparação dos Dados - Tratamento de dados)
# --------------------------------------------------------
@timer_decorator
def prepare_data(conn):
    """Prepara e limpa os dados."""
    df = pd.read_sql_query("""
        SELECT 
            o.order_id, o.customer_id, o.order_status,
            o.order_purchase_timestamp, o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            i.price, i.freight_value, i.seller_id,
            p.product_category_name,
            c.customer_unique_id, c.customer_state,
            s.seller_state,
            r.review_score
        FROM orders o
        INNER JOIN order_items i ON o.order_id = i.order_id
        INNER JOIN products p ON i.product_id = p.product_id
        INNER JOIN customers c ON o.customer_id = c.customer_id
        INNER JOIN sellers s ON i.seller_id = s.seller_id
        LEFT JOIN order_reviews r ON o.order_id = r.order_id
    """, conn)

    # Limpeza básica
    df = df.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])
    df = df[df['price'] > 0]
    df = df[df['freight_value'] >= 0]
    
    # Preenche valores nulos de review_score com a mediana
    df['review_score'] = df['review_score'].fillna(df['review_score'].median())

    # Conversão de datas
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

    # Features básicas
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

    return df

# Executa a preparação dos dados
df = prepare_data(conn)

# --------------------------------------------------------
# Criação de novas features a partir de timestamps
# --------------------------------------------------------
# Dia da semana (1=Segunda, ..., 7=Domingo)
df['day_of_week_int'] = df['order_purchase_timestamp'].dt.weekday + 1
# Mês do pedido (YYYY-MM)
df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
# Tempo de entrega em dias
df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# =============================================================================
# 2. ANÁLISE EXPLORATÓRIA DE DADOS
# =============================================================================
# Esta seção responde perguntas fundamentais sobre o negócio:
# - Volume de pedidos mensais e sazonalidade
# - Distribuição do tempo de entrega
# - Relação entre frete e distância 
# - Categorias mais vendidas por faturamento
# - Estados com maior valor médio de pedido
# =============================================================================

@timer_decorator
def analyze_monthly_orders(df):
    """
    Analisa o volume de pedidos por mês.
    """
    monthly_orders = df.groupby('order_month')['order_id'].nunique().reset_index(name='order_count')
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=monthly_orders, x='order_month', y='order_count', 
                marker='o', color=COLORS['primary'], linewidth=2)
    plt.title('Volume de Pedidos por Mês', color=COLORS['text'], pad=20)
    plt.xlabel('Mês', color=COLORS['text'])
    plt.ylabel('Número de Pedidos', color=COLORS['text'])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_file(plt.gcf(), 'volume_pedidos_mes.png')

@timer_decorator
def analyze_delivery_time(df):
    """
    Analisa a distribuição do tempo de entrega.
    """
    plt.figure(figsize=(12, 8))
    sns.histplot(df['delivery_time'], bins=30, kde=True, 
                color=COLORS['primary'], alpha=0.6)
    plt.title('Distribuição do Tempo de Entrega (dias)', color=COLORS['text'], pad=20)
    plt.xlabel('Dias', color=COLORS['text'])
    plt.ylabel('Frequência', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'distribuicao_tempo_entrega.png')

@timer_decorator
def analyze_freight_distance(df):
    """
    Analisa a relação entre valor do frete e distância de entrega.
    """
    # Agrupa por estado do cliente e vendedor
    freight_by_state = df.groupby(['customer_state', 'seller_state']).agg({
        'freight_value': 'mean',
        'price': 'mean'
    }).reset_index()
    
    # Calcula uma distância aproximada baseada em estados diferentes
    freight_by_state['is_same_state'] = freight_by_state['customer_state'] == freight_by_state['seller_state']
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=freight_by_state, x='is_same_state', y='freight_value',
                palette=[COLORS['primary'], COLORS['secondary']])
    plt.title('Valor do Frete por Mesmo Estado', color=COLORS['text'], pad=20)
    plt.xlabel('Mesmo Estado', color=COLORS['text'])
    plt.ylabel('Valor do Frete (R$)', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'frete_vs_estado.png')
    
    # Análise adicional: frete vs valor do pedido
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=freight_by_state, x='price', y='freight_value', 
                   alpha=0.6, color=COLORS['primary'])
    plt.title('Relação entre Valor do Pedido e Frete', color=COLORS['text'], pad=20)
    plt.xlabel('Valor Médio do Pedido (R$)', color=COLORS['text'])
    plt.ylabel('Valor do Frete (R$)', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'frete_vs_valor.png')

@timer_decorator
def analyze_top_categories(df):
    """
    Analisa as categorias de produtos mais vendidas em faturamento.
    """
    cat_sales = df.groupby('product_category_name')['price'].sum().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=cat_sales.head(10), x='price', y='product_category_name',
                palette='viridis')
    plt.title('Top 10 Categorias por Faturamento', color=COLORS['text'], pad=20)
    plt.xlabel('Faturamento (R$)', color=COLORS['text'])
    plt.ylabel('Categoria', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'top10_categorias_faturamento.png')

@timer_decorator
def analyze_state_orders(df):
    """
    Analisa os estados com maior valor médio de pedido.
    """
    order_values = df.groupby(['order_id', 'customer_state'])['price'].sum().reset_index()
    state_avg = order_values.groupby('customer_state')['price'].mean().sort_values(ascending=False).reset_index()

    if not state_avg.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(data=state_avg.head(10), x='price', y='customer_state',
                    palette='viridis')
        plt.title('Top 10 Estados por Valor Médio de Pedido', color=COLORS['text'], pad=20)
        plt.xlabel('Valor Médio (R$)', color=COLORS['text'])
        plt.ylabel('Estado', color=COLORS['text'])
        plt.grid(True, alpha=0.3)
        save_file(plt.gcf(), 'top10_estados_valor_medio.png')

# Executa as análises exploratórias
analyze_monthly_orders(df)
analyze_delivery_time(df)
analyze_freight_distance(df)
analyze_top_categories(df)
analyze_state_orders(df)

# =============================================================================
# 3. SOLUÇÃO DE PROBLEMAS DE NEGÓCIO
# =============================================================================
# Esta seção implementa soluções para problemas específicos de negócio:
# - Análise de retenção de clientes e identificação de padrões de compra
# - Predição de atraso na entrega usando aprendizado de máquina
# - Segmentação de clientes usando técnicas de clustering (RFM)
# - Análise detalhada da satisfação do cliente e fatores de impacto
# =============================================================================

@timer_decorator
def analyze_customer_retention(df):
    """
    Analisa a taxa de retenção de clientes.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
    """
    # Agrupa por customer_unique_id para identificar clientes únicos
    customer_orders = df.groupby('customer_unique_id').agg({
        'order_id': 'nunique',
        'order_purchase_timestamp': ['min', 'max']
    }).reset_index()
    
    customer_orders.columns = ['customer_unique_id', 'order_count', 'first_order', 'last_order']
    
    # Calcula métricas
    total_customers = len(customer_orders)
    recurring_customers = len(customer_orders[customer_orders['order_count'] > 1])
    retention_rate = recurring_customers / total_customers
    
    print(f"\nAnálise de Retenção de Clientes:")
    print(f"Total de clientes únicos: {total_customers}")
    print(f"Clientes recorrentes: {recurring_customers}")
    print(f"Taxa de retenção: {retention_rate:.2%}")
    
    # Distribuição de pedidos por cliente
    plt.figure(figsize=(12, 8))
    ax = sns.histplot(data=customer_orders, x='order_count', bins=20, 
                     color=COLORS['primary'], kde=True, alpha=0.7)
    plt.title('Distribuição de Pedidos por Cliente', color=COLORS['text'], pad=20)
    plt.xlabel('Número de Pedidos', color=COLORS['text'])
    plt.ylabel('Quantidade de Clientes', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    # Adicionar textos informativos no gráfico
    plt.text(0.95, 0.95, f'Total de Clientes: {total_customers}\nClientes Recorrentes: {recurring_customers}\nTaxa de Retenção: {retention_rate:.2%}',
             transform=ax.transAxes, fontsize=12, verticalalignment='top', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))
    save_file(plt.gcf(), 'distribuicao_pedidos_cliente.png')
    
    # Análise adicional: tempo entre primeiro e último pedido
    customer_orders['days_between_orders'] = (customer_orders['last_order'] - customer_orders['first_order']).dt.days
    
    plt.figure(figsize=(12, 8))
    sns.histplot(data=customer_orders[customer_orders['order_count'] > 1], 
                x='days_between_orders', bins=30, 
                color=COLORS['primary'], kde=True, alpha=0.7)
    plt.title('Distribuição do Tempo entre Primeiro e Último Pedido', color=COLORS['text'], pad=20)
    plt.xlabel('Dias entre Pedidos', color=COLORS['text'])
    plt.ylabel('Quantidade de Clientes', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'tempo_entre_pedidos.png')

@timer_decorator
def predict_delivery_delay(df):
    """
    Implementa modelo de predição de atraso na entrega usando XGBoost.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
    """
    # Features para o modelo
    features = [
        'price', 'freight_value', 'delivery_time',
        'product_weight_g', 'product_length_cm',
        'product_height_cm', 'product_width_cm',
        'payment_installments'
    ]

    # Verifica se todas as features existem e não têm valores nulos
    available_features = [f for f in features if f in df.columns]
    print(f"\nFeatures disponíveis para o modelo: {available_features}")

    # Remove registros com valores nulos nas features
    df_model = df.dropna(subset=available_features)

    X = df_model[available_features]
    y = df_model['is_late']

    # Balanceamento das classes
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Treina modelo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Avaliação do modelo
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.6f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nImportância das features:")
    print(feature_importance.round(6))

    # Plota curva ROC
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, color=COLORS['primary'], linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.fill_between(fpr, tpr, alpha=0.2, color=COLORS['primary'])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'curva_roc.png')

    # Plota feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Importância das Features')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'feature_importance.png')

@timer_decorator
def segment_customers(df):
    """
    Realiza segmentação de clientes usando clustering.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
    """
    # Cálculo de RFM (Recência, Frequência, Valor)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (df['order_purchase_timestamp'].max() - x.max()).days,
        'order_id': 'nunique',  # Conta pedidos únicos
        'price': 'sum',
        'review_score': 'mean'
    }).reset_index()
    rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary', 'satisfaction']

    # Remove outliers do RFM usando método mais eficiente
    for col in ['recency', 'frequency', 'monetary', 'satisfaction']:
        Q1 = rfm[col].quantile(0.25)
        Q3 = rfm[col].quantile(0.75)
        IQR = Q3 - Q1
        rfm = rfm[(rfm[col] >= Q1 - 1.5 * IQR) & (rfm[col] <= Q3 + 1.5 * IQR)]

    # Normalização
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary', 'satisfaction']])

    # KMeans com número fixo de clusters para otimização
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Sumário de clusters
    cluster_summary = rfm.groupby('cluster').agg({
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'],
        'monetary': ['mean', 'median'],
        'satisfaction': ['mean', 'median'],
        'customer_unique_id': 'count'
    }).round(6)
    
    print("\nSumário dos clusters:")
    print(cluster_summary)

    # Gráfico de dispersão otimizado
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(rfm['recency'], rfm['monetary'], 
                         c=rfm['cluster'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Segmentação de Clientes (RFM)')
    plt.xlabel('Recência (dias)')
    plt.ylabel('Monetário (R$)')
    plt.grid(True)
    save_file(plt.gcf(), 'segmentacao_clientes_rfm.png')

    # Análise adicional dos clusters
    print("\nCaracterísticas dos Clusters:")
    for cluster in range(n_clusters):
        cluster_data = rfm[rfm['cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Tamanho: {len(cluster_data)} clientes")
        print(f"Recência média: {cluster_data['recency'].mean():.2f} dias")
        print(f"Frequência média: {cluster_data['frequency'].mean():.2f} pedidos")
        print(f"Valor médio: R$ {cluster_data['monetary'].mean():.2f}")
        print(f"Satisfação média: {cluster_data['satisfaction'].mean():.2f}")
        
        # Análise de distribuição de frequência
        freq_dist = cluster_data['frequency'].value_counts().sort_index()
        print("\nDistribuição de Frequência:")
        for freq, count in freq_dist.items():
            print(f"{freq} pedido(s): {count} clientes")

@timer_decorator
def analyze_customer_satisfaction(df):
    """
    Analisa a satisfação dos clientes.
    """
    # Satisfação vs tempo de entrega
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='review_score', y='delivery_time',
                palette='viridis')
    plt.title('Satisfação vs Tempo de Entrega', color=COLORS['text'], pad=20)
    plt.xlabel('Nota de Avaliação', color=COLORS['text'])
    plt.ylabel('Dias de Entrega', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'satisfacao_vs_tempo_entrega.png')

    # Satisfação vs valor do pedido
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='review_score', y='price',
                palette='viridis')
    plt.title('Satisfação vs Valor do Pedido', color=COLORS['text'], pad=20)
    plt.xlabel('Nota de Avaliação', color=COLORS['text'])
    plt.ylabel('Preço (R$)', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'satisfacao_vs_valor_pedido.png')

    # Análise estatística
    satisfaction_stats = df.groupby('review_score').agg({
        'delivery_time': ['mean', 'median', 'std'],
        'price': ['mean', 'median', 'std'],
        'order_id': 'count'
    }).round(6)
    
    print("\nEstatísticas de Satisfação:")
    print(satisfaction_stats)

    # Distribuição das notas
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='review_score', palette='viridis')
    plt.title('Distribuição das Notas de Avaliação', color=COLORS['text'], pad=20)
    plt.xlabel('Nota de Avaliação', color=COLORS['text'])
    plt.ylabel('Quantidade de Avaliações', color=COLORS['text'])
    plt.grid(True, alpha=0.3)
    save_file(plt.gcf(), 'distribuicao_avaliacoes.png')

# Executa as análises de negócio
analyze_customer_retention(df)
predict_delivery_delay(df)
segment_customers(df)
analyze_customer_satisfaction(df)

# =============================================================================
# 4. VISUALIZAÇÃO E DASHBOARDS
# =============================================================================
# Esta seção cria visualizações interativas dos resultados:
# - Dashboard geral com evolução de vendas ao longo do tempo 
# - Mapa de calor mostrando vendas por estado/região
# - Visualizações da relação entre satisfação e tempo de entrega
# - Dashboard de análise de desempenho de vendedores
# =============================================================================

@timer_decorator
def create_consolidated_dashboard(df):
    """Cria um dashboard consolidado com todos os gráficos."""
    
    # Cores personalizadas para o dashboard
    colors = {
        'bg': '#003057',  # Azul escuro
        'text': '#FFFFFF',  # Branco
        'grid': '#31446B',  # Azul médio
        'highlight': '#F15A29'  # Laranja (destaque)
    }
    
    # Template personalizado para o Plotly
    template_personalizado = go.layout.Template()
    template_personalizado.layout.paper_bgcolor = colors['bg']
    template_personalizado.layout.plot_bgcolor = colors['bg']
    template_personalizado.layout.font = dict(color=colors['text'])
    template_personalizado.layout.title = dict(font=dict(color=colors['text']))
    template_personalizado.layout.colorway = ["#F15A29", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f"]
    
    # Criar dashboard de vendas por estado com mapa
    # Analisar vendas por estado
    state_orders = df[df['customer_state'].notna()].groupby('customer_state').agg({
        'order_id': 'nunique',
        'price': 'sum',
        'freight_value': 'mean',
        'review_score': 'mean'
    }).reset_index()
    
    # Adicionar informação de região
    state_region_map = {
        'AC': 'Norte', 'AM': 'Norte', 'AP': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste',
        'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }
    state_orders['region'] = state_orders['customer_state'].map(state_region_map)
    
    # Ordenar por valor total de vendas
    state_sales_total = state_orders.sort_values('price', ascending=False)
    
    # Criar tabela com dados por estado
    tabela_estados = go.Figure(data=[go.Table(
        header=dict(
            values=['Estado', 'Região', 'Vendas (R$)', 'Qtd. Pedidos', 'Frete Médio (R$)', 'Avaliação Média'],
            line_color='darkslategray',
            fill_color=colors['highlight'],
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                state_sales_total['customer_state'],
                state_sales_total['region'],
                state_sales_total['price'].apply(lambda x: f'R$ {x:,.2f}'),
                state_sales_total['order_id'],
                state_sales_total['freight_value'].apply(lambda x: f'R$ {x:,.2f}'),
                state_sales_total['review_score'].apply(lambda x: f'{x:.2f}')
            ],
            line_color='darkslategray',
            fill_color=[[colors['bg'] if i % 2 == 0 else '#001e3c' for i in range(len(state_sales_total))]],
            align='center',
            font=dict(color='white', size=11)
        )
    )])
    
    tabela_estados.update_layout(
        template=template_personalizado,
        height=600,
        width=600,
        margin=dict(t=0, b=0, l=0, r=0),
    )
    
    # Criar mapa de vendas por estado
    fig_map = go.Figure()
    
    # Adicionar mapa de calor para os estados
    fig_map.add_trace(go.Heatmap(
        z=state_sales_total['price'],
        x=state_sales_total['customer_state'],
        y=[1]*len(state_sales_total),  # Todos na mesma linha
        colorscale='Blues',
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Vendas (R$)",
                side="right",
                font=dict(size=14, color="white")
            ),
            tickfont=dict(color="white"),
            x=1.02
        ),
        hovertemplate='<b>Estado: %{x}</b><br>Vendas: R$ %{z:,.2f}<br>Região: %{customdata}<extra></extra>',
        customdata=state_sales_total['region']
    ))
    
    fig_map.update_layout(
        template=template_personalizado,
        height=500,
        width=1200,
        margin=dict(t=120, b=80, l=80, r=80),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color="white"),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        xaxis=dict(
            title=dict(
                text="Estados",
                font=dict(color="white", size=14)
            ),
            tickfont=dict(color="white", size=12),
            tickmode="array",
            tickvals=state_sales_total['customer_state'],
            ticktext=state_sales_total['customer_state']
        )
    )
    
    # Criar dashboard de vendas por mês, categoria e tempo
    
    # Análise de vendas por mês
    monthly_orders = df.copy()
    monthly_orders['order_month'] = pd.to_datetime(monthly_orders['order_purchase_timestamp']).dt.strftime('%Y-%m')
    monthly_sales = monthly_orders.groupby('order_month').agg({
        'order_id': 'nunique',
        'price': 'sum'
    }).reset_index()
    
    # Top 10 categorias mais vendidas
    category_sales = df.groupby('product_category_name').agg({
        'order_id': 'nunique',
        'price': 'sum'
    }).reset_index()
    top_categories = category_sales.sort_values('price', ascending=False).head(10)
    
    # Tempo médio de entrega por mês
    monthly_orders['delivery_time'] = (pd.to_datetime(monthly_orders['order_delivered_customer_date']) - 
                                      pd.to_datetime(monthly_orders['order_purchase_timestamp'])).dt.days
    avg_delivery_time = monthly_orders.groupby('order_month')['delivery_time'].mean().reset_index()
    
    # Top 10 vendedores por faturamento
    seller_sales = df.groupby('seller_id').agg({
        'order_id': 'nunique',
        'price': 'sum',
        'review_score': 'mean'
    }).reset_index()
    top_sellers = seller_sales.sort_values('price', ascending=False).head(10)
    
    # Criação da figura principal com 6 subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        subplot_titles=(
            "Vendas por Mês", "Vendas por Estado (Top 10)", 
            "Top 10 Categorias (Faturamento)", "Tempo Médio de Entrega por Mês",
            "Top 10 Vendedores (Faturamento)", "Satisfação por Vendedor (Top 10)"
        )
    )
    
    # 1. Vendas por mês
    fig.add_trace(
        go.Bar(
            x=monthly_sales['order_month'],
            y=monthly_sales['price'],
            marker_color=colors['highlight'],
            hovertemplate='<b>%{x}</b><br>Vendas: R$ %{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Vendas por estado (top 10)
    top_states = state_orders.sort_values('price', ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=top_states['customer_state'],
            y=top_states['price'],
            marker_color=colors['highlight'],
            hovertemplate='<b>%{x}</b><br>Vendas: R$ %{y:,.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Top 10 categorias (faturamento)
    fig.add_trace(
        go.Bar(
            x=top_categories['price'],
            y=top_categories['product_category_name'],
            orientation='h',
            marker_color=colors['highlight'],
            hovertemplate='<b>%{y}</b><br>Vendas: R$ %{x:,.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Tempo médio de entrega por mês
    fig.add_trace(
        go.Scatter(
            x=avg_delivery_time['order_month'],
            y=avg_delivery_time['delivery_time'],
            mode='lines+markers',
            marker=dict(color=colors['highlight']),
            line=dict(color=colors['highlight']),
            hovertemplate='<b>%{x}</b><br>Tempo médio: %{y:.1f} dias<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Top 10 vendedores (faturamento)
    fig.add_trace(
        go.Bar(
            x=top_sellers['seller_id'].apply(lambda x: x[:6] + '...'),
            y=top_sellers['price'],
            marker_color=colors['highlight'],
            hovertemplate='<b>%{x}</b><br>Vendas: R$ %{y:,.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Satisfação por vendedor (top 10)
    fig.add_trace(
        go.Bar(
            x=top_sellers['seller_id'].apply(lambda x: x[:6] + '...'),
            y=top_sellers['review_score'],
            marker_color=colors['highlight'],
            hovertemplate='<b>%{x}</b><br>Nota média: %{y:.2f}<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Configuração dos layouts para os 6 gráficos principais
    fig.update_layout(
        template=template_personalizado,
        height=900,
        width=1400,
        margin=dict(t=150, b=20, l=50, r=50),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        showlegend=False,
        
        # Definir os 6 subplots
        xaxis1=dict(
            domain=[0.05, 0.45], anchor="y1",
            title=dict(text="Mês", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        yaxis1=dict(
            domain=[0.55, 0.78], anchor="x1",
            title=dict(text="Vendas (R$)", font=dict(color="white")),
            tickfont=dict(color="white"), tickformat=",.2f", gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        
        xaxis2=dict(
            domain=[0.55, 0.95], anchor="y2",
            title=dict(text="Estado", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        yaxis2=dict(
            domain=[0.55, 0.78], anchor="x2",
            title=dict(text="Vendas (R$)", font=dict(color="white")),
            tickfont=dict(color="white"), tickformat=",.2f", gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        
        xaxis3=dict(
            domain=[0.05, 0.45], anchor="y3",
            title=dict(text="Vendas (R$)", font=dict(color="white")),
            tickfont=dict(color="white"), tickformat=",.2f", gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        yaxis3=dict(
            domain=[0.22, 0.45], anchor="x3",
            title=dict(text="Categoria", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        
        xaxis4=dict(
            domain=[0.55, 0.95], anchor="y4",
            title=dict(text="Mês", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        yaxis4=dict(
            domain=[0.22, 0.45], anchor="x4",
            title=dict(text="Dias", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        
        xaxis5=dict(
            domain=[0.05, 0.45], anchor="y5",
            title=dict(text="Vendedor", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        yaxis5=dict(
            domain=[0.02, 0.15], anchor="x5",
            title=dict(text="Vendas (R$)", font=dict(color="white")),
            tickfont=dict(color="white"), tickformat=",.2f", gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        
        xaxis6=dict(
            domain=[0.55, 0.95], anchor="y6",
            title=dict(text="Vendedor", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
        yaxis6=dict(
            domain=[0.02, 0.15], anchor="x6",
            title=dict(text="Nota", font=dict(color="white")),
            tickfont=dict(color="white"), gridcolor=colors['grid'],
            showgrid=True, zeroline=False
        ),
    )
    
    # Adicionar título principal
    fig.add_annotation(
        x=0.5, y=1.12,
        xref="paper", yref="paper",
        text="DASHBOARD OLIST E-COMMERCE",
        font=dict(size=26, color=colors['text']),
        showarrow=False
    )
    
    fig.add_annotation(
        x=0.5, y=1.07,
        xref="paper", yref="paper",
        text="Análise de vendas e desempenho da plataforma brasileira de e-commerce",
        font=dict(size=16, color=colors['text']),
        showarrow=False
    )
    
    # Salvar os dois dashboards em arquivos separados
    try:
        fig.write_html(os.path.join(RESULTS_PATH, 'dashboard_vendas.html'))
        fig_map.write_html(os.path.join(RESULTS_PATH, 'dashboard_mapa.html'))
        tabela_estados.write_html(os.path.join(RESULTS_PATH, 'tabela_estados.html'))
        print("Dashboards interativos salvos com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar dashboards interativos: {e}")
    
    # Criar um dashboard consolidado com links para os outros dashboards
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard Olist - Análise Completa de E-commerce</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #003057;
                color: white;
            }}
            .container {{
                width: 100%;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                background-color: #001e3c;
                border-radius: 5px;
            }}
            .dashboard-container {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .dashboard {{
                width: 100%;
                height: 900px;
                border: none;
                border-radius: 5px;
            }}
            .row {{
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .col {{
                flex: 1;
            }}
            .map-dashboard {{
                width: 100%;
                height: 600px;
                border: none;
                border-radius: 5px;
            }}
            .tabela-dashboard {{
                width: 100%;
                height: 600px;
                border: none;
                border-radius: 5px;
            }}
            h1, h2, h3 {{
                color: white;
            }}
            .card {{
                background-color: #001e3c;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section-title {{
                background-color: #F15A29;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                margin-top: 40px;
                margin-bottom: 20px;
                font-weight: bold;
            }}
            .description {{
                margin-bottom: 20px;
                line-height: 1.6;
            }}
            .requirements {{
                background-color: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .requirements ul {{
                margin: 0;
                padding-left: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>DASHBOARD OLIST E-COMMERCE</h1>
                <p>Análise de dados e insights estratégicos da plataforma brasileira de e-commerce</p>
            </div>
            
            <div class="description">
                Este dashboard apresenta a análise completa do dataset de e-commerce brasileiro da Olist,
                seguindo os requisitos estabelecidos no desafio técnico. A análise está dividida em seções principais,
                conforme solicitado no escopo do projeto.
            </div>
            
            <div class="section-title">1. VISÃO GERAL DE VENDAS</div>
            
            <div class="requirements">
                <p><strong>Requisito atendido:</strong> "Um dashboard geral que mostre a evolução das vendas ao longo do tempo"</p>
                <ul>
                    <li>Evolução de vendas mensais</li>
                    <li>Análise por estado e categoria</li>
                    <li>Tempo médio de entrega</li>
                    <li>Desempenho de vendedores</li>
                </ul>
            </div>
            
            <div class="card">
                <h2>Dashboard Geral Evolutivo</h2>
                <iframe class="dashboard" src="dashboard_vendas.html"></iframe>
            </div>
            
            <div class="section-title">2. DISTRIBUIÇÃO GEOGRÁFICA DAS VENDAS</div>
            
            <div class="requirements">
                <p><strong>Requisito atendido:</strong> "Um mapa de calor mostrando a concentração de vendas por região/estado do Brasil"</p>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>Mapa de Vendas por Estado</h2>
                        <iframe class="map-dashboard" src="dashboard_mapa.html"></iframe>
                    </div>
                </div>
                <div class="col">
                    <div class="card">
                        <h2>Dados Detalhados por Estado</h2>
                        <iframe class="tabela-dashboard" src="tabela_estados.html"></iframe>
                    </div>
                </div>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    try:
        with open(os.path.join(RESULTS_PATH, 'dashboard_consolidado.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard consolidado salvo em {os.path.join(RESULTS_PATH, 'dashboard_consolidado.html')}")
    except Exception as e:
        print(f"Erro ao salvar dashboard consolidado: {e}")
    
    return fig

# Substitui as chamadas individuais por uma única chamada
create_consolidated_dashboard(df)

# --------------------------------------------------------
# Encerramento
# --------------------------------------------------------
# Fecha conexão com o banco de dados
conn.close()

print("\nAnálise concluída com sucesso!")
print(f"Resultados salvos em: {RESULTS_PATH}")
