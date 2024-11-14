from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# 백엔드를 'Agg'로 설정하여 GUI 창을 띄우지 않고 이미지만 생성
plt.switch_backend('Agg')

app = Flask(__name__)

# 다중 항목 데이터 저장소
inventory = {}

# 피드백 데이터 저장소
feedback_data = {
    "adjustment_history": pd.DataFrame(columns=["date", "daily_usage", "feedback", "adjustment_factor"])
}

# 홈 페이지 라우트
@app.route('/')
def home():
    return render_template('index.html', inventory=inventory)

# 항목 추가 및 업데이트 처리
@app.route('/add_item', methods=['POST'])
def add_item():
    item_name = request.form['item_name']
    current_stock = float(request.form['current_stock'])
    daily_usage = float(request.form['daily_usage'])
    unit = request.form['unit']
    alert_days_before = int(request.form['alert_days_before'])
    
    # 항목 데이터 초기화 및 저장
    inventory[item_name] = {
        "current_stock": current_stock,
        "daily_usage": daily_usage,
        "unit": unit,
        "last_update": datetime.now().strftime('%Y-%m-%d'),
        "alert_days_before": alert_days_before,
        "consumption_history": pd.DataFrame({
            'ds': pd.date_range(start=datetime.now().strftime('%Y-%m-%d'), periods=10, freq='D'),
            'y': [daily_usage] * 10
        })
    }
    
    return redirect(url_for('home'))

# 예측 및 그래프 생성
@app.route('/predict/<item_name>')
def predict(item_name):
    if item_name not in inventory:
        return redirect(url_for('home'))
    
    item = inventory[item_name]
    consumption_history = item["consumption_history"]
    
    # Prophet 모델 학습
    model = Prophet()
    model.fit(consumption_history)
    
    # 미래 예측
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # 소모 예상 시점 계산
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
    
    # Gradient Update on Daily Average Usage 그래프 생성
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
    
    return render_template('predict.html', item_name=item_name, item=item, graph_url=graph_url, usage_graph_url=usage_graph_url)

# 피드백 처리 라우트
@app.route('/feedback/<item_name>', methods=['POST'])
def feedback(item_name):
    feedback = request.form['feedback']
    feedback_date = datetime.strptime(request.form['feedback_date'], '%Y-%m-%d')
    item = inventory.get(item_name)
    
    if not item:
        return redirect(url_for('home'))
    
    # Gradient Descent 업데이트 규칙
    adjustment_factor = 0.05  # 학습률
    adjustment = adjustment_factor * (1 if feedback == "late" else -1)
    item["daily_usage"] += adjustment
    item["daily_usage"] = max(0.1, item["daily_usage"])  # 최소 사용량 제한

    # 피드백 데이터 기록
    new_feedback_row = {
        "date": feedback_date,
        "daily_usage": item["daily_usage"],
        "feedback": feedback,
        "adjustment_factor": adjustment
    }
    feedback_data["adjustment_history"] = pd.concat([feedback_data["adjustment_history"], pd.DataFrame([new_feedback_row])], ignore_index=True)
    
    return redirect(url_for('predict', item_name=item_name))

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
    alert_date = predicted_empty_date - timedelta(days=item["alert_days_before"])
    
    # 업데이트된 값 반영
    item["predicted_empty_date"] = predicted_empty_date
    item["alert_date"] = alert_date
    
    return redirect(url_for('predict', item_name=item_name))



if __name__ == '__main__':
    app.run(debug=True)

