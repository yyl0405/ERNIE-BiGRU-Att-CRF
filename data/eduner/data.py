import csv

# 定义输入的CSV文件路径和输出的TXT文件路径
csv_file_path = 'test.csv'
txt_file_path = 'test.txt'

# 打开CSV文件并读取内容
with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
    # 创建CSV阅读器对象
    csvreader = csv.reader(csvfile)
    
    # 打开或创建TXT文件准备写入
    with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
        # 遍历CSV文件的每一行
        for row in csvreader:
            # 将每一行转换为字符串，并写入到TXT文件中
            txtfile.write(','.join(row) + '\n')

print("CSV 文件已成功转换为 TXT 文件！")