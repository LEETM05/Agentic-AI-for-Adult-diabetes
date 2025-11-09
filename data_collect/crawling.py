import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import trafilatura
from fpdf import FPDF
import chardet
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font("NotoSansKR", "", "NotoSansKR-Regular.ttf", uni=True)
        self.add_font('NotoSansKR', 'B', 'NotoSansKR-Bold.ttf', uni=True)
        self.set_font("NotoSansKR", size=12)

    def add_rich_text(self, tagged_lines):
        for tag, text in tagged_lines:
            if tag == 'h1':
                self.set_font("NotoSansKR", style="B", size=18)
            elif tag == 'h2':
                self.set_font("NotoSansKR", style="B", size=16)
            elif tag == 'h3':
                self.set_font("NotoSansKR", style="B", size=14)
            else:
                self.set_font("NotoSansKR", size=12)
            self.multi_cell(0, 8, text)
            self.ln(1)

    def add_table_as_key_value(self, table_data):
        if len(table_data) < 2:
            return
        headers = table_data[0]
        for idx, row in enumerate(table_data[1:], start=1):
            self.set_font("NotoSansKR", style="B", size=12)
            self.cell(0, 10, f"연번: {idx}", ln=True)
            self.set_font("NotoSansKR", size=11)
            for key, value in zip(headers, row):
                self.multi_cell(0, 8, f"{key.strip()}: {value.strip()}")
            self.ln(3)

def normalize_table(table):
    result = []
    span_map = {}

    rows = table.find_all('tr')
    for row_idx in range(len(rows)):
        tr = rows[row_idx]
        row = []
        col_idx = 0
        cells = tr.find_all(['td', 'th'])
        cell_cursor = 0

        while cell_cursor < len(cells) or (row_idx in span_map and span_map[row_idx]):
            if row_idx in span_map and (col_idx in [s[0] for s in span_map[row_idx]]):
                for s in span_map[row_idx]:
                    if s[0] == col_idx:
                        row.append(s[1])
                        span_map[row_idx].remove(s)
                        break
                col_idx += 1
                continue

            if cell_cursor >= len(cells):
                break

            cell = cells[cell_cursor]
            text = cell.get_text(strip=True)
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    for offset in range(1, rowspan):
                        span_map.setdefault(row_idx + offset, []).append((col_idx, text))
                col_idx += 1
            cell_cursor += 1

        result.append(row)
    return result

def extract_text_with_headings(soup):
    contents = []
    blacklist = ["advertisement", "advertisements", "sponsored", "sponsor", "powered by"]
    
    def is_advertising(text):
        lowered = text.lower()
        return any(bad_word in lowered for bad_word in blacklist)

    # for elem in soup.find_all(['h1', 'h2', 'h3', 'p']):
    for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'li']):
        text = elem.get_text(strip=True)
        if text and not is_advertising(text):
            contents.append((elem.name, text))
    return contents

def save_text_as_pdf(tagged_text, tables, filename, url):
    pdf = PDF()
    pdf.add_rich_text(tagged_text)
    if tables:
        pdf.ln(5)
        pdf.set_font("NotoSansKR", style="B", size=12)
        pdf.cell(0, 10, "※ 표 정보", ln=True)
        pdf.set_font("NotoSansKR", size=11)
        pdf.add_table_as_key_value(tables)
    pdf.ln(10)
    pdf.set_font("NotoSansKR", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 10, f"출처 : {url}")
    pdf.output(filename)

# URL 목록
with open('urls.txt', 'r', encoding='utf-8') as f:
    urls = [line.strip() for line in f if line.strip()]

output_dir = 'crawled_pdfs'
os.makedirs(output_dir, exist_ok=True)

# 크롤링 및 PDF 저장 루프
for i, url in enumerate(tqdm(urls, desc="크롤링 및 PDF 저장")):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        detected = chardet.detect(response.content)
        response.encoding = detected['encoding'] if detected['encoding'] else 'utf-8'

        if response.status_code != 200:
            print(f"[{i}] 실패: {url} (status {response.status_code})")
            continue

        # 1. trafilatura 기반 본문 추출
        downloaded = trafilatura.fetch_url(url)
        plain_text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)

        # 2. 구조 기반 제목 정보 추출
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        tagged_text = extract_text_with_headings(soup)
        print(f"[{i}] 구조 태그 수:", [(tag, len(list(soup.find_all(tag)))) for tag in ['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'li']])

        # 구조 정보가 부족하면 fallback to plain text
        if not tagged_text or len(tagged_text) < 5:
            tagged_text = [('p', line) for line in plain_text.splitlines() if line.strip()]

        # 3. 표 추출
        tables = []
        for table in soup.find_all('table'):
            normalized = normalize_table(table)
            if normalized:
                tables = normalized
                break

        if not tagged_text:
            print(f"[{i}] 본문 추출 실패: {url}")
            continue

        filename = os.path.join(output_dir, f"site_{i+1:03}.pdf")
        save_text_as_pdf(tagged_text, tables, filename, url)

    except Exception as e:
        print(f"[{i}] 예외 발생: {url}\n{e}")