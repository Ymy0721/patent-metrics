import pandas as pd

def convert_excel_to_csv(excel_path: str, csv_path: str):
    # 使用 openpyxl 作为引擎，并可根据情况指定 header 参数
    df = pd.read_excel(excel_path, engine='openpyxl')  # 如数据错位，可尝试添加 header 参数：header=0 或 header=None
    # 保存为 CSV 文件，指定编码为 utf-8-sig 防止乱码
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
if __name__ == '__main__':
    # Excel 文件路径和生成的 CSV 文件路径
    excel_file = './data/raw/patents.xlsx'
    csv_file = './data/raw/patents.csv'
    convert_excel_to_csv(excel_file, csv_file)
    print(f"已将 {excel_file} 转换为 {csv_file}")
