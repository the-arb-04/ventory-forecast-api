import pandas as pd
from prophet import Prophet
from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def create_forecast():
    try:
        payload = request.get_json()
        sales_data = payload.get('salesData')
        horizon = payload.get('horizon', 60)  # Default 2 months forecast

        if not sales_data or len(sales_data) < 10:
            return jsonify({"error": "Not enough data provided. At least 10 data points are required for a reliable forecast."}), 400

        # Create DataFrame
        df = pd.DataFrame(sales_data)
        df = df.rename(columns={'saleTimestamp': 'ds', 'quantitySold': 'y'})
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

        # Fill in missing days with zero sales
        full_dates = pd.date_range(df['ds'].min(), df['ds'].max(), freq='D')
        df = df.set_index('ds').reindex(full_dates, fill_value=0).reset_index()
        df.columns = ['ds', 'y']

        # Build Prophet model without daily seasonality
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        model.fit(df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=horizon, freq='D')
        forecast = model.predict(future)

        # Keep only required fields for the forecast period
        prediction_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
        prediction_data['ds'] = prediction_data['ds'].apply(lambda x: x.isoformat())

        return jsonify(prediction_data.to_dict('records'))

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return jsonify({
            "error": "An internal error occurred in the Python service.",
            "traceback": error_trace
        }), 500

if __name__ == '__main__':
    # The port will be provided by the deployment environment
    app.run(debug=False, host='0.0.0.0', port=5000)
