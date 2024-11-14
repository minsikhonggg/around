from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

# 백엔드를 'Agg'로 설정하여 GUI 창을 띄우지 않고 이미지만 생성
plt.switch_backend('Agg')

app = Flask(__name__)

# 다중 항목 데이터 저장소
inventory = {}

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
        "alert_days_before": alert_days_before,  # 알림 시점 설정
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

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label="Predicted Usage")
    plt.axvline(predicted_empty_date, color='red', linestyle='--', label=f"Out of Stock: {predicted_empty_date.date()}")
    plt.axvline(alert_date, color='blue', linestyle='--', label=f"Alert Date: {alert_date.date()}")
    plt.xlabel('Date')
    plt.ylabel(f'Expected Usage ({item["unit"]})')
    plt.title(f'{item_name} Consumption Forecast')
    plt.legend()
    
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # 업데이트된 예측 정보 전달
    return render_template('predict.html', item_name=item_name, item=item, empty_date=predicted_empty_date.date(), alert_date=alert_date.date(), graph_url=graph_url)

# 피드백 처리 라우트
@app.route('/feedback/<item_name>', methods=['POST'])
def feedback(item_name):
    feedback = request.form['feedback']
    feedback_date = datetime.strptime(request.form['feedback_date'], '%Y-%m-%d')
    item = inventory.get(item_name)
    
    if not item:
        return redirect(url_for('home'))
    
    # 피드백에 따른 Daily Average Usage 조정
    if feedback == "early":
        adjustment_factor = 0.9  # 알림을 더 늦게 보냄 (소모량 줄임)
    elif feedback == "late":
        adjustment_factor = 1.1  # 알림을 더 빨리 보냄 (소모량 늘림)
    else:
        adjustment_factor = 1.0  # 아무 변화 없음
    
    # 해당 피드백 날짜에 맞춰 소비 이력에 반영
    if feedback_date in item["consumption_history"]['ds'].values:
        item["consumption_history"].loc[item["consumption_history"]['ds'] == feedback_date, 'y'] *= adjustment_factor
    else:
        # 만약 피드백 날짜가 기존 소비 이력에 없을 경우 추가
        new_row = pd.DataFrame({'ds': [feedback_date], 'y': [item["daily_usage"] * adjustment_factor]})
        item["consumption_history"] = pd.concat([item["consumption_history"], new_row]).sort_values(by='ds').reset_index(drop=True)
    
    # 새로운 일일 소모량을 계산하여 평균값 업데이트
    item["daily_usage"] = item["consumption_history"]['y'].mean()  # 새로운 평균 daily_usage로 업데이트
    
    # Prophet 모델 재학습 (소비 이력에 반영된 최신 데이터를 학습)
    item["model"] = Prophet()
    item["model"].fit(item["consumption_history"])
    
    return redirect(url_for('predict', item_name=item_name))

# 구매 이벤트 처리 라우트
@app.route('/purchase/<item_name>', methods=['POST'])
def purchase(item_name):
    item = inventory.get(item_name)
    
    if not item:
        return redirect(url_for('home'))
    
    # 구매 후 재고 업데이트
    purchase_quantity = float(request.form['purchase_quantity'])
    item["current_stock"] += purchase_quantity
    
    return redirect(url_for('predict', item_name=item_name))

if __name__ == '__main__':
    app.run(debug=True)
