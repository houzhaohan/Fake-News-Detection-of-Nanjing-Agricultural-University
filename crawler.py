# -*- coding: utf-8 -*-
# 爬虫

import requests
from bs4 import BeautifulSoup
import time

# 更新为正确的新闻列表URL
base_url = "https://news.njau.edu.cn"
urls = [f"{base_url}/nnyw/list{i}.htm" for i in range(1, 75)]  # 先测试5页

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": base_url
}

all_titles = []

for url in urls:
    print(f"\n正在爬取: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")

        # 尝试自动检测编码
        if response.encoding == 'ISO-8859-1':
            response.encoding = 'utf-8'

        print(f"实际编码: {response.encoding}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # 调试：保存网页内容供检查
            with open('debug_page.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("已保存网页内容到debug_page.html")

            # 尝试多种选择器
            selectors = [
                'div.news_list ul li',  # 可能的实际选择器
                'ul.news_list li',  # 另一种可能
                'div.list ul li a',  # 第三种可能
                'div.article-list li'  # 第四种可能
            ]

            found_items = None
            for selector in selectors:
                items = soup.select(selector)
                if items:
                    print(f"使用选择器 '{selector}' 找到 {len(items)} 个元素")
                    found_items = items
                    break

            if found_items:
                for item in found_items:
                    try:
                        title = item.text.strip()
                        if len(title) > 5:  # 简单过滤无效标题
                            all_titles.append(title)
                            print(f"找到标题: {title}")
                    except Exception as e:
                        print(f"处理单个项目时出错: {e}")
            else:
                print("警告: 未找到新闻项目，请检查选择器")

        else:
            print(f"请求失败，状态码: {response.status_code}")

        time.sleep(3)  # 增加延迟

    except Exception as e:
        print(f"爬取过程中出错: {str(e)}")

# 保存结果
if all_titles:
    with open('njau_news_titles.txt', 'w', encoding='utf-8') as f:
        for i, title in enumerate(all_titles, 1):
            f.write(f"{i}. {title}\n")
    print(f"\n成功爬取 {len(all_titles)} 条新闻标题，已保存到 njau_news_titles.txt")
else:
    print("\n未爬取到任何新闻标题，请检查调试信息")


# import re
#
#
# def remove_dates_from_file(input_file, output_file):
#     # 正则表达式匹配行末尾的日期（YYYY-MM-DD格式）
#     date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}$')
#
#     with open(input_file, 'r', encoding='utf-8') as infile, \
#             open(output_file, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             # 删除行末尾的日期
#             cleaned_line = date_pattern.sub('', line).rstrip()
#             # 写入新文件
#             outfile.write(cleaned_line + '\n')
#
#
# # 使用示例
# input_filename = 'njau_news_titles.txt'  # 替换为你的输入文件名
# output_filename = 'njau_news.txt'  # 输出文件名
# remove_dates_from_file(input_filename, output_filename)
#
# print(f"日期已从 {input_filename} 中删除，结果保存在 {output_filename}")