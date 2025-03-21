# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ================== 全局配置 ==================
NUM_PROCESSES = 7  # 设置进程数

# ================== 数据预处理 ==================
def preprocess_dates(df):
    df['First_Date'] = df['Application Date'].astype(str).str.split(';').str[0]
    df['Clean_Date'] = df['First_Date'].str.replace(r'[^0-9.]', '', regex=True)
    df['Application Date'] = pd.to_datetime(df['Clean_Date'], format='%Y.%m.%d', errors='coerce')
    invalid_count = df['Application Date'].isna().sum()
    if invalid_count > 0:
        print(f"{invalid_count} useless data records have been deleted")
    df = df.dropna(subset=['Application Date']).reset_index(drop=True)
    df['Year'] = df['Application Date'].dt.year
    return df.drop(columns=['First_Date', 'Clean_Date'])

# ================== 多进程构建实体库 ==================
def load_stop_words():
    # 读取 stop_words.txt，如果文件存在，则加载停用词
    stop_words = set()
    if os.path.exists("stop_words.txt"):
        with open("stop_words.txt", "r", encoding="utf-8") as f:
            stop_words = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(stop_words)} stop words.")
    else:
        print("No stop_words.txt found. Skipping stop word filtering.")
    return stop_words

def process_entities_chunk(chunk, entity_count, entity_years, lock):
    index_range, entities_list, years = chunk
    local_count = {}
    local_years = {}
    
    for i, entities in enumerate(entities_list):
        if not entities:
            continue
        for entity in entities:
            if entity not in local_count:
                local_count[entity] = 0
                local_years[entity] = set()
            local_count[entity] += 1
            local_years[entity].add(years[i])
    
    with lock:
        for entity, count in local_count.items():
            entity_count[entity] = entity_count.get(entity, 0) + count
            entity_years[entity] = entity_years.get(entity, set()).union(local_years[entity])
    
    return len(index_range)

def build_entity_library(df):
    # 检查是否已存在 entity_library.json 文件
    if os.path.exists('entity_library.json'):
        print("Found existing entity_library.json. Loading it...")
        with open('entity_library.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    stop_words = load_stop_words()
    entities_list = df['Extracted Entities'].str.split('; ').tolist()
    years = df['Year'].tolist()
    
    manager = mp.Manager()
    entity_count = manager.dict()
    entity_years = manager.dict()
    lock = manager.Lock()
    
    num_records = len(df)
    chunk_size = (num_records + NUM_PROCESSES - 1) // NUM_PROCESSES
    chunks = [
        (
            range(i * chunk_size, min((i + 1) * chunk_size, num_records)), 
            entities_list[i * chunk_size: min((i + 1) * chunk_size, num_records)], 
            years[i * chunk_size: min((i + 1) * chunk_size, num_records)]
        ) 
        for i in range(NUM_PROCESSES)
    ]
    
    with tqdm(total=num_records, desc="? Extracting entities...") as pbar:
        with mp.Pool(NUM_PROCESSES) as pool:
            for processed in pool.imap_unordered(partial(process_entities_chunk, 
                                                         entity_count=entity_count, 
                                                         entity_years=entity_years, 
                                                         lock=lock), chunks):
                pbar.update(processed)
    
    final_entity_lib = {
        e: {"count": int(entity_count[e]), "years": sorted(entity_years[e])} 
        for e in entity_count.keys() if e not in stop_words
    }
    
    sorted_entity_lib = dict(sorted(final_entity_lib.items(), key=lambda item: item[1]['count'], reverse=True))
    
    with open('entity_library.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_entity_lib, f, ensure_ascii=False, indent=2)
    
    return sorted_entity_lib
# ================== 多进程指标计算 ==================
def calculate_metrics_batch(args):
    indices, tfidf, entities_list, entity_lib = args
    batch_metrics = []
    for idx in indices:
        ents = entities_list[idx]
        # 1. 新颖性（TF-IDF向量平均值）
        novelty = np.mean(tfidf[idx].toarray())
        
        # 2. 行业趋势（实体年度变化率）
        trend_scores = []
        for e in ents:
            years = entity_lib.get(e, {}).get('years', [])
            if len(years) < 2:
                trend_scores.append(0)
            else:
                slope = linregress(range(len(years)), years).slope
                trend_scores.append(slope)
        trend = np.mean(trend_scores) if trend_scores else 0
        
        # 3. 适用范围
        applicability_scores = []
        for e in ents:
            # 筛选包含该实体的所有文档索引
            indices_e = [i for i, doc in enumerate(entities_list) if e in doc]
            if len(indices_e) < 2:
                applicability_scores.append(0)
            else:
                tfidf_subset = tfidf[indices_e]
                # 根据样本数量确定聚类数（至少1个，最多3个）
                n_clusters = 3 if len(indices_e) >= 3 else len(indices_e)
                try:
                    kmeans_e = KMeans(n_clusters=n_clusters, random_state=42).fit(tfidf_subset)
                    cluster_labels_e = kmeans_e.labels_
                    cluster_dist_e = pd.Series(cluster_labels_e).value_counts(normalize=True)
                    applicability_e = 1 - np.sum(cluster_dist_e.pow(2))
                except Exception as ex:
                    applicability_e = 0
                applicability_scores.append(applicability_e)
        applicability_value = np.mean(applicability_scores) if applicability_scores else 0
        
        # 4. 配套技术依存度（共现次数）
        co_occur = 0
        current_set = set(ents)
        for other in entities_list:
            co_occur += len(current_set & set(other))
        co_occur -= len(ents)  # 排除自身计数
        
        # 5. 可替代性（余弦相似度）
        sims = cosine_similarity(tfidf[idx:idx+1], tfidf).flatten()
        if len(sims) > 1:
            replaceability = np.mean(np.delete(sims, idx))
        else:
            replaceability = 0
        
        # 6. 成熟度（频率 * 时间跨度）
        maturity_sum = 0
        for e in ents:
            info = entity_lib.get(e, {})
            freq = info.get('count', 0)
            years_list = info.get('years', [])
            if years_list:
                span = max(years_list) - min(years_list) + 1
            else:
                span = 0
            maturity_sum += freq * span
        maturity = maturity_sum / len(ents) if ents else 0
        
        batch_metrics.append([novelty, trend, applicability_value, co_occur, replaceability, maturity])
    # 返回元组：(本批次处理的记录数, 批次结果)
    return (len(indices), batch_metrics)

def parallel_metrics_calculation(df, entities_list, entity_lib):
    # 生成文档（将每个记录的实体列表拼接为字符串）并计算全局TF-IDF矩阵
    documents = [' '.join(ents) for ents in entities_list]
    tfidf = TfidfVectorizer().fit_transform(documents)
    
    num_records = len(df)
    indices = np.arange(num_records)
    index_chunks = np.array_split(indices, NUM_PROCESSES)
    
    tasks = [(chunk, tfidf, entities_list, entity_lib) for chunk in index_chunks]
    
    all_metrics = []
    with tqdm(total=num_records, desc="Calculating metrics") as pbar:
        with mp.Pool(NUM_PROCESSES) as pool:
            for processed, batch_result in pool.imap_unordered(calculate_metrics_batch, tasks):
                all_metrics.extend(batch_result)
                pbar.update(processed)
            pool.close()
            pool.join()
    
    # 如果返回结构为嵌套列表则进行合并（这里已是展平后的列表）
    metrics = all_metrics
    return metrics

# ================== 熵权法计算权重 ==================
def entropy_weight(data):
    epsilon = 1e-12
    p = data / (np.sum(data, axis=0) + epsilon)
    entropy = -np.sum(p * np.log(p + epsilon), axis=0)
    weights = (1 - entropy) / np.sum(1 - entropy)
    return weights

# ================== 主程序 ==================
if __name__ == '__main__':
    # 读取数据
    df = pd.read_excel('extracted_entities.xlsx', sheet_name='Sheet1')
    df = preprocess_dates(df)
    
    # 构建实体库（多进程）
    entity_lib = build_entity_library(df)
    print(f"Entity Number: {len(entity_lib)}")
    
    # 准备数据：将“Extracted Entities”列按分号分割为列表
    entities_list = df['Extracted Entities'].str.split('; ').tolist()
    
    # 多进程计算指标
    metrics = parallel_metrics_calculation(df, entities_list, entity_lib)
    
    # 归一化处理
    scaler = MinMaxScaler()
    metrics_norm = scaler.fit_transform(metrics)
    
    # 使用熵权法计算各指标权重
    weights = entropy_weight(metrics_norm)
    indicator_names = ['Novelty', 'Trend', 'Applicability', 'Dependency', 'Replaceability', 'Maturity']
    print("The weights of each indicator:", dict(zip(indicator_names, np.round(weights, 3))))
    
    # 将各指标加入DataFrame
    df[indicator_names] = metrics_norm
    
    # 计算综合得分
    df['Score'] = np.dot(metrics_norm, weights)
    
    # 按综合得分降序排序，并保存结果到Excel
    df = df.sort_values('Score', ascending=False)
    df.to_excel('output_with_metrics.xlsx', index=False)
    
    print("Result have been saved to output_with_metrics.xlsx")
