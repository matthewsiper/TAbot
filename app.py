import os
from datetime import datetime, timedelta, timezone

from config import CONFIG_PARAMS
from chart_builder_worker import ChartBuilderWorker
from constants import BASE_MODELS_DIR, TICKERS_FOR_DROPDOWN, PERIODICITY, CHART_PATTERN, PRED_DICT, P_CACHE

import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import boto3
import argparse
import pika
import json


conn = boto3.client(
    's3',
    aws_access_key_id='AKIAIWPOHHSFY2F4UEOA',
    aws_secret_access_key='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'
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

parser = argparse.ArgumentParser(description='UI / Master node.')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=8223)


AWS_DOMAIN = CONFIG_PARAMS["aws_pub_domain"]
CHART_BUILDER_WORKER = ChartBuilderWorker()
MODELS = [tf.keras.models.load_model(f'{BASE_MODELS_DIR}models/model{model_num}/model{model_num}.h5') for model_num in ["1","2","3"]]


def get_predictions(a_ticker, path="/Users/mattscomputer/6513 P2/live/{}_1.jpg"):
    try:
        path = path.format(ticker) if path == "/Users/mattscomputer/6513 P2/live/{}_1.jpg" else \
            path + f"/{a_ticker}_1.jpg"
        img = image.load_img(path, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = {}

        for model, model_name in zip(MODELS, ["model1", "model2", "model3"]):
            _image = np.vstack([x])
            num_label = np.argmax(model.predict(x), axis=-1)[0]
            label_prob = model.predict(_image, batch_size=1)[0][num_label]
            class_label = PRED_DICT[num_label]
            preds[model_name] = (label_prob, class_label)

        os.system(f"rm -f {path}")
        return preds
    except:
        msg = "Error in get_predictions"
        raise Exception(msg)


def get_price_data(ticker, periodicity=None):
    try:
        if periodicity is None or not isinstance(periodicity, str):
            msg = "periodicity must be str type"
            raise Exception(msg)
        t = yf.Ticker(ticker)
        hist = t.history(period=periodicity)
        hist.index.rename("time", inplace=True)
        return hist
    except:
        msg = "Error occurred in get_price_data method"
        raise Exception(msg)

def generate_table(pred_dict):
    return dt.DataTable(
        id='datatable',
        columns=[{"name": i, "id": i} for i in pred_dict.columns],
        data=pred_dict.to_dict(orient="records"),
     )

def to_unix_microseconds(dt):
    global utcTimeZone
    epoch = datetime.fromtimestamp(0, tz=utcTimeZone)
    unixTimeStamp = (dt - epoch).total_seconds()
    return unixTimeStamp * 1000


def utc_milleseconds_to_date(microseconds):
    global utcTimeZone
    seconds = microseconds / 1000
    parsedDateTime = datetime.fromtimestamp(seconds)
    d_aware = parsedDateTime.astimezone(utcTimeZone)
    return d_aware


def findPriceExtreme(a_df, startTime, endTime):
    minRowIndex = getRowIndexOfTime(startTime)
    maxRowIndex = getRowIndexOfTime(endTime)
    sl = slice(minRowIndex, maxRowIndex + 1)
    viewBarData = a_df.iloc[sl]
    maxBarHigh = max(viewBarData['High'])
    minBarLow = min(viewBarData['Low'])

    return [minBarLow, maxBarHigh]


def getRowIndexOfTime(currentDateTime):
    global df
    global dateformat

    roundedDateTime = roundToMinutes(currentDateTime)
    indexTimeString = roundedDateTime.strftime(dateformat)
    rowIndex = np.where(df.index == indexTimeString)[0]

    return rowIndex[0]


def roundToMinutes(tm):
    barPeriod = 1
    roundedDateTime = tm.replace(second=0, microsecond=0)
    minuteValue = roundedDateTime.minute
    differenceMinutes = minuteValue % barPeriod
    minuteValue = minuteValue - differenceMinutes
    roundedDateTime = roundedDateTime.replace(minute=minuteValue)

    return roundedDateTime


def keepInXAxisBounds(value):
    global minTimeStamp
    global maxTimeStamp
    adjustedValue = 0

    if (value > maxTimeStamp):
        adjustedValue = maxTimeStamp
    elif (value < minTimeStamp):
        adjustedValue = minTimeStamp
    else:
        adjustedValue = value
    return adjustedValue


def getChartBarNumbers():
    global df
    chartBarNumbers = []
    for index, row in df.iterrows():
        xValue = row['time']
        yValue = row['low'] - .0001
        annotation = dict(
            x=xValue, y=yValue, xref='x', yref='y',
            showarrow=False, xanchor='center',
            text=index, font=dict(
                family='Arial',
                size=10,
                color='#000000'
            )
        )
        chartBarNumbers.append(annotation)
    return chartBarNumbers


def convert_preds_to_data_table(pred_dict):
    preds = {"Model":[], "Probability":[], "Prediction": []}

    for model_name, pred_tuple in pred_dict.items():
        preds["Probability"].append(pred_tuple[0])
        preds["Prediction"].append(pred_tuple[1])
        preds["Model"].append(model_name)
    return pd.DataFrame(data=preds, columns=[i for i in list(preds.keys())])


for ticker in TICKERS_FOR_DROPDOWN:
    df = get_price_data(ticker["value"], PERIODICITY)
    CHART_BUILDER_WORKER.build_chart(CHART_PATTERN, df, file_num=1, ticker=ticker["value"],
                                     alternate_path="/Users/mattscomputer/6513 P2/cache")


df = get_price_data(TICKERS_FOR_DROPDOWN[0]["value"], PERIODICITY)
CHART_BUILDER_WORKER.build_chart(CHART_PATTERN, df, file_num=1, ticker=TICKERS_FOR_DROPDOWN[0]["value"])

PREDICTIONS_CACHE = dict([(ticker["value"],
                           get_predictions(ticker["value"],
                                           path="/Users/mattscomputer/6513 P2/cache")) for
                          ticker in TICKERS_FOR_DROPDOWN])

df = get_price_data(TICKERS_FOR_DROPDOWN[0]["value"], PERIODICITY)
CHART_BUILDER_WORKER.build_chart(CHART_PATTERN, df, file_num=1, ticker=TICKERS_FOR_DROPDOWN[0]["value"])

opens = {"Open": []}
closes = {"Close": []}
highs = {"High": []}
lows = {"Low": []}


utcTimeZone = timezone.utc
chartShift = 14400 * 1000
dateformat = '%Y-%m-%d %H:%M:%S'

rowLimit = 500
df = df[:rowLimit]

dateMin = min(df.index)
dateMax = max(df.index)

dateMin = datetime.strptime(str(dateMin), dateformat)
dateMax = datetime.strptime(str(dateMax), dateformat)

dateMin = dateMin.replace(tzinfo=utcTimeZone)
dateMax = dateMax.replace(tzinfo=utcTimeZone)

minTimeStamp = to_unix_microseconds(dateMin)
maxTimeStamp = to_unix_microseconds(dateMax)

minTimeStampChart = minTimeStamp + chartShift
maxTimeStampChart = maxTimeStamp + chartShift
sliderStep = timedelta(minutes=1).total_seconds() * 1000
tickPadding = .0010

startingPriceExtreme = findPriceExtreme(df, dateMin, dateMax)
newRangeLow = round(startingPriceExtreme[0], 4) - tickPadding
newRangeHigh = round(startingPriceExtreme[1], 4) + tickPadding

firstTrace = go.Candlestick(
    x=df.index,
    open=df.Open,
    high=df.High,
    low=df.Low,
    close=df.Close
)

data = [firstTrace]

layout = go.Layout(
    xaxis=dict(
        autorange=False,
        rangeslider=dict(
            visible=False
        ),
        type='date',
        range=[minTimeStampChart, maxTimeStampChart]
    ),
    yaxis=dict(
        title='Ticks',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='lightgrey'
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='all',
        tickmode='linear',
        tick0=0.500,
        dtick=.0001,
        autorange=False,
        range=[newRangeLow, newRangeHigh]

    ),
    height=500,
    width=1100,
)

figure = go.Figure(data=data, layout=layout)

graph = dcc.Graph(
    id='graph-with-slider',
    figure=figure
)

app.layout = html.Div([
    html.Div([
        html.Label('Select ticker:'),
        dcc.Dropdown(id='ticker_dropdown',
                     options=TICKERS_FOR_DROPDOWN,
                     value=TICKERS_FOR_DROPDOWN[0]["value"],
                     style={"width": 200},
        ),
        html.Div([html.Button('Train', id='submit-val', n_clicks=0)]),
        html.Div(id="container-button-basic"),
        html.Div([graph]),
        dt.DataTable(id='datatable', columns=[{"name": i, "id": i} for i in (
            convert_preds_to_data_table(P_CACHE[TICKERS_FOR_DROPDOWN[0]["value"]])).columns],
                     data=(convert_preds_to_data_table(P_CACHE[TICKERS_FOR_DROPDOWN[0]["value"]])).to_dict(
                         orient="records"),
                     editable=True),
        html.Label('X Range Slider'),
        dcc.RangeSlider(
            id='x-range-slider',
            min=minTimeStamp,
            max=maxTimeStamp,
            step=sliderStep,
            value=[minTimeStamp, maxTimeStamp],
            allowCross=False,
            updatemode='mouseup'
        )]
    )
])


@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')])
def issue_train(n_clicks):
    print(f"inside of issue_train")
    start = str(datetime.now())
    for q in ["queue1", "queue2", "queue3"]:
        channel.basic_publish(exchange='cnn_exchange',
                      routing_key=q,
                      body=json.dumps({"message": "train", "time": start}))
    return


@app.callback(dash.dependencies.Output('datatable', 'data'),
      [dash.dependencies.Input('ticker_dropdown', 'value')])
def update_table(ticker):
    if "{}.jpg".format(ticker) not in os.listdir('/Users/mattscomputer/6513 P2/live/'):
        CHART_BUILDER_WORKER.build_chart(CHART_PATTERN, df, file_num=1, ticker=ticker)
    preds = convert_preds_to_data_table(P_CACHE[ticker])
    data = preds.to_dict(orient="records")
    return data


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
     [dash.dependencies.Input('ticker_dropdown', 'value'), dash.dependencies.Input('x-range-slider', 'value')])
def adjust_ranges(ticker, xSliderValue):
    global data
    global layout
    global tickPadding

    df = get_price_data(ticker, PERIODICITY)

    firstTrace = go.Candlestick(
        x=df.index,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close)

    data = [firstTrace]

    xMinAdjusted = keepInXAxisBounds(xSliderValue[0])
    xMaxAdjusted = keepInXAxisBounds(xSliderValue[1])

    xminDate = utc_milleseconds_to_date(xMinAdjusted)
    xmaxDate = utc_milleseconds_to_date(xMaxAdjusted)

    xminChartValue = to_unix_microseconds(xminDate) + chartShift
    xmaxChartValue = to_unix_microseconds(xmaxDate) + chartShift

    newYRange = findPriceExtreme(df, xminDate, xmaxDate)
    newRangeLow = round(newYRange[0], 4) - tickPadding
    newRangeHigh = round(newYRange[1], 4) + tickPadding

    layout['xaxis']['range'] = [xminChartValue, xmaxChartValue]
    layout['yaxis']['range'] = [newRangeLow, newRangeHigh]

    figure = go.Figure(data=data, layout=layout)

    return figure


if __name__ == '__main__':
    args = parser.parse_args()
    host = args.host
    port = args.port
    params = pika.ConnectionParameters(host='localhost',
                                       heartbeat=600,
                                       blocked_connection_timeout=300)

    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.exchange_declare(exchange='cnn_exchange',
                             exchange_type='direct')

    for queue in ["queue1", "queue2", "queue3"]:
        channel.queue_declare(queue=queue, exclusive=False)
        channel.queue_bind(
            exchange='cnn_exchange', queue=queue, routing_key=queue)
    app.run_server(host, port)
