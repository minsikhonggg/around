from flask import Flask, render_template, request, redirect, url_for
import pytesseract
from PIL import Image
import io
import pandas as pd
import re
import os

app = Flask(__name__)

# 영수증 데이터 저장소 (예시로 pandas DataFrame 사용)
receipt_data = pd.DataFrame(columns=["Item", "Price", "Quantity", "Date"])

# 업로드된 영수증을 처리하는 함수
def extract_receipt_info(image):
    # 이미지 파일을 PIL로 열기
    img = Image.open(image)
    
    # OCR을 사용하여 텍스트 추출
    extracted_text = pytesseract.image_to_string(img)
    
    # 영수증에서 필요한 정보 추출 (예시: 가격, 물품 이름 등)
    items = []
    prices = []
    quantities = []
    date = None
    
    # 간단한 정규식 예시 (가격, 품목 이름, 수량 등 추출)
    item_pattern = r"([a-zA-Z\s]+)\s+(\d+\.\d{2})"  # 물품 이름과 가격
    date_pattern = r"\d{4}-\d{2}-\d{2}"  # 날짜 형식 예시 (YYYY-MM-DD)
    
    # 날짜 추출
    date_match = re.search(date_pattern, extracted_text)
    if date_match:
        date = date_match.group(0)
    
    # 물품과 가격 추출
    item_matches = re.findall(item_pattern, extracted_text)
    for match in item_matches:
        items.append(match[0].strip())
        prices.append(float(match[1]))
        quantities.append(1)  # 예시로 수량을 1로 설정 (나중에 확장 가능)
    
    # 추출된 데이터 반환
    return items, prices, quantities, date

# 물품 등록 및 데이터베이스 업데이트
@app.route('/upload_receipt', methods=['GET', 'POST'])
def upload_receipt():
    if request.method == 'POST':
        # 이미지 파일 받기
        file = request.files['receipt_image']
        
        if file:
            # 영수증 정보 추출
            items, prices, quantities, date = extract_receipt_info(file)
            
            # DataFrame에 데이터 추가
            global receipt_data
            for item, price, quantity in zip(items, prices, quantities):
                receipt_data = receipt_data.append({
                    "Item": item,
                    "Price": price,
                    "Quantity": quantity,
                    "Date": date
                }, ignore_index=True)
            
            # 데이터베이스(혹은 CSV 등) 저장 (선택적)
            receipt_data.to_csv("receipt_data.csv", index=False)
            
            return redirect(url_for('view_receipts'))
    
    return render_template('upload_receipt.html')

# 영수증 목록을 보여주는 페이지
@app.route('/view_receipts')
def view_receipts():
    global receipt_data
    return render_template('view_receipts.html', receipts=receipt_data.to_html())

if __name__ == '__main__':
    app.run(debug=True)
