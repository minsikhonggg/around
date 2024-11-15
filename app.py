# Flask 웹 애플리케이션
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# 'Agg' 백엔드 설정: GUI 창 없이 이미지 생성
plt.switch_backend('Agg')

app = Flask(__name__)

# 항목별 재고 및 소모량을 저장하는 딕셔너리
inventory = {}

# 피드백 기록을 저장하는 데이터프레임
feedback_data = {
    "adjustment_history": pd.DataFrame(columns=["date", "daily_usage", "feedback", "adjustment_factor"])
}

# 홈 페이지 라우트: 사용자에게 항목과 알림을 보여주는 페이지
@app.route('/')
def home():
    return render_template('index.html', inventory=inventory)

# 항목 추가 및 업데이트 처리: 사용자가 새 항목을 추가하는 라우트
@app.route('/add_item', methods=['POST'])
def add_item():
    item_name = request.form['item_name']
    current_stock = float(request.form['current_stock'])
    daily_usage = float(request.form['daily_usage'])
    unit = request.form['unit']
    alert_days_before = int(request.form['alert_days_before'])
    
    # 항목 데이터 초기화 및 저장
    predicted_days_left = current_stock / daily_usage  # 예상 소진까지 남은 일수 계산
    predicted_empty_date = datetime.now() + timedelta(days=predicted_days_left)  # 예상 소진 날짜 계산
    alert_date = predicted_empty_date - timedelta(days=alert_days_before)  # 알림 날짜 계산 (소진 날짜에서 미리 알림 날짜만큼 빼기)
 
    # 항목 데이터 저장
    inventory[item_name] = {
        "current_stock": current_stock,
        "daily_usage": daily_usage,
        "unit": unit,
        "last_update": datetime.now().strftime('%Y-%m-%d'),
        "alert_days_before": alert_days_before,
        "alert_date": alert_date,
        "consumption_history": pd.DataFrame({
            'ds': pd.date_range(start=datetime.now().strftime('%Y-%m-%d'), periods=10, freq='D'),
            'y': [daily_usage] * 10  # 소비 이력 데이터 (10일간 동일한 소비량)
        })
    }
    
    return redirect(url_for('home'))

# 예측 및 그래프 생성: 예측 결과를 바탕으로 소비 예측 및 그래프를 생성하는 라우트
@app.route('/predict/<item_name>')
def predict(item_name):
    if item_name not in inventory:
        return redirect(url_for('home'))
    
    item = inventory[item_name]
    consumption_history = item["consumption_history"]
    
    # Prophet 모델을 사용한 예측
    model = Prophet()
    model.fit(consumption_history)  # 소비 이력 데이터를 기반으로 모델 학습
    
    # 미래 예측 (30일 예측)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # 소모 예상 시점 계산 (현재 재고와 소비량을 기반으로)
    predicted_days_left = item["current_stock"] / item["daily_usage"]
    predicted_empty_date = datetime.now() + timedelta(days=predicted_days_left)
    alert_date = predicted_empty_date - timedelta(days=item["alert_days_before"])

    # 예측 그래프 생성 (소비 예측에 Daily Average Usage 반영)
    plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'] * item["daily_usage"] / forecast['yhat'].mean(), label="Predicted Usage (Adjusted)")
    plt.axvline(predicted_empty_date, color='red', linestyle='--', label=f"Out of Stock: {predicted_empty_date.date()}")
    plt.axvline(alert_date, color='blue', linestyle='--', label=f"Alert Date: {alert_date.date()}")
    plt.xlabel('Date')
    plt.ylabel(f'Expected Usage ({item["unit"]})')
    plt.title(f'{item_name} Consumption Forecast')
    plt.legend()
    
    # 예측 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Gradient Update on Daily Average Usage 그래프 생성 (소모량 변화를 시각화)
    if not feedback_data["adjustment_history"].empty:
        adjustment_history_df = feedback_data["adjustment_history"]
        plt.figure(figsize=(10, 6))
        plt.plot(adjustment_history_df['date'], adjustment_history_df['daily_usage'], marker='o', linestyle='-', color='green', label="Daily Average Usage")
        plt.xlabel('Date')
        plt.ylabel('Daily Usage')
        plt.title(f'{item_name} Gradient Update on Daily Average Usage')
        plt.legend()
        plt.grid(True)

        # 그래프를 이미지로 변환
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        usage_graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
    else:
        usage_graph_url = None
    
    return render_template('predict.html', item_name=item_name, item=item, graph_url=graph_url, usage_graph_url=usage_graph_url, empty_date=predicted_empty_date.date(), alert_date=alert_date.date())

# 피드백 처리: 사용자가 알림 날짜와 소모량에 대해 피드백을 주는 라우트
@app.route('/feedback/<item_name>', methods=['POST'])
def feedback(item_name):
    feedback = request.form['feedback']
    feedback_date = datetime.strptime(request.form['feedback_date'], '%Y-%m-%d')
    item = inventory.get(item_name)
    
    if not item:
        return redirect(url_for('home'))
    
    # 피드백에 따른 알림 날짜 및 소모량 조정
    if feedback == "late":
        # "Too late" 피드백: 알림 날짜 하루 당기기
        item["alert_date"] += timedelta(days=1)
        
        # 소모량 증가: 하루 소비량 증가 (예시로 10% 증가)
        adjustment = item["daily_usage"] * 0.1  # 10% 증가
        item["daily_usage"] += adjustment
        
        # "Too late" 피드백 수 증가
        item['too_late_feedback_count'] = item.get('too_late_feedback_count', 0) + 1
    
    elif feedback == "early":
        # "Too early" 피드백: 알림 날짜 하루 미루기
        item["alert_date"] -= timedelta(days=1)
        
        # 소모량 감소: 하루 소비량 감소 (예시로 10% 감소)
        adjustment = item["daily_usage"] * -0.1  # 10% 감소
        item["daily_usage"] += adjustment
        
        # "Too early" 피드백 수 증가
        item['too_early_feedback_count'] = item.get('too_early_feedback_count', 0) + 1
    
    # 소모량 변경 후 소진 날짜 및 알림 날짜 갱신
    predicted_days_left = item["current_stock"] / item["daily_usage"]
    predicted_empty_date = datetime.now() + timedelta(days=predicted_days_left)
    
    # 알림 날짜는 소진 날짜에서 미리 알림 날짜만큼 빼서 설정
    alert_date = predicted_empty_date - timedelta(days=item["alert_days_before"])
    
    # 업데이트된 값 반영
    item["predicted_empty_date"] = predicted_empty_date
    item["alert_date"] = alert_date  # 알림 날짜 재계산
    
    # 피드백 데이터 기록
    item['feedback_count'] = item.get('feedback_count', 0) + 1  # 피드백 횟수 증가
    
    new_feedback_row = {
        "date": feedback_date,
        "daily_usage": item["daily_usage"],
        "feedback": feedback,
        "adjustment_factor": adjustment
    }
    
    feedback_data["adjustment_history"] = pd.concat([feedback_data["adjustment_history"], pd.DataFrame([new_feedback_row])], ignore_index=True)
    
    return redirect(url_for('predict', item_name=item_name))

# 구매 처리: 사용자가 재고를 추가하고 알림 날짜를 갱신하는 라우트
@app.route('/purchase/<item_name>', methods=['POST'])
def purchase(item_name):
    item = inventory.get(item_name)
    
    if not item:
        return redirect(url_for('home'))
    
    # 구매 후 재고 업데이트
    purchase_quantity = float(request.form['purchase_quantity'])
    item["current_stock"] += purchase_quantity
    
    # 재고가 업데이트된 후, 예상 소진 날짜 및 알림 날짜를 재계산
    predicted_days_left = item["current_stock"] / item["daily_usage"]
    predicted_empty_date = datetime.now() + timedelta(days=predicted_days_left)
    
    # 만약 재고가 추가된 후, 예상 소진 날짜가 이미 지나면 알림을 취소
    if predicted_empty_date > item["alert_date"]:
        item["alert_date"] = None  # 알림 취소
    
    alert_date = predicted_empty_date - timedelta(days=item["alert_days_before"])
    
    # 업데이트된 값 반영
    item["predicted_empty_date"] = predicted_empty_date
    item["alert_date"] = alert_date
    
    return redirect(url_for('predict', item_name=item_name))

# 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)
