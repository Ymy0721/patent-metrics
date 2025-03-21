import pandas as pd
import jieba
from collections import Counter

def count_tf(text_series):
    counter = Counter()
    for text in text_series.dropna():
        tokens = jieba.lcut(text)
        counter.update(tokens)
    return counter

def main():
    # Read CSV file containing patents data
    df = pd.read_csv('./data/raw/patents.csv')
    
    # Compute term frequencies for 发明名称 (title) and 摘要 (abstract)
    title_counter = count_tf(df['发明名称'])
    abstract_counter = count_tf(df['摘要'])
    
    # Get sorted term frequencies (high to low)
    title_common = title_counter.most_common()
    abstract_common = abstract_counter.most_common()
    
    # Print the term with the highest frequency for each
    if title_common:
        title_highest = title_common[0]
        print("Title highest frequency term:", title_highest)
    else:
        print("No terms found in 发明名称 column.")
    
    if abstract_common:
        abstract_highest = abstract_common[0]
        print("Abstract highest frequency term:", abstract_highest)
    else:
        print("No terms found in 摘要 column.")
    
    # Save the sorted term frequencies to files
    with open('./data/stat/title_tf.txt', 'w', encoding='utf-8') as f:
        for term, freq in title_common:
            f.write(f"{term}\t{freq}\n")
    
    with open('./data/stat/abstract_tf.txt', 'w', encoding='utf-8') as f:
        for term, freq in abstract_common:
            f.write(f"{term}\t{freq}\n")

if __name__ == '__main__':
    main()