from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/forecast')
def forecast_template():
    revenue_data = pd.read_pickle("./models/revenue_data.pkl")
    income_data = pd.read_pickle("./models/income_data.pkl")

    # Reset the index to ensure 'Quarter' is a column
    revenue_data.reset_index(inplace=True)
    income_data.reset_index(inplace=True)

    # Merge the two dataframes on 'Quarter'
    merged_data = pd.merge(revenue_data, income_data, on='Quarter')

    # Create a ColumnDataSource from the merged data
    source = ColumnDataSource(data=merged_data)

    # Create a new plot with a datetime axis type
    p = figure(x_axis_type="datetime", title="Revenue and Income Over Time", height=470, width=1260)
    p.title.text_font_size = '15pt'
    p.title.text_font = 'cursive'
    p.title.text_align = 'right'

    # Add the revenue line renderer with legend and line thickness
    revenue_line = p.line(x="Quarter", y="Revenue (US $M)", source=source, line_width=2,
                          color="blue", legend_label="Revenue")
    p.circle(x="Quarter", y="Revenue (US $M)", source=source, color="blue")

    # Configure the primary y-axis
    p.yaxis.axis_label = "Revenue (US $M)"
    p.yaxis.axis_line_color = "blue"
    p.yaxis.major_label_text_color = "blue"
    p.yaxis.major_tick_line_color = "blue"
    p.yaxis.minor_tick_line_color = "green"

    # Add a secondary y-axis for income data
    p.extra_y_ranges = {"income": Range1d(start=merged_data["Net Income (US $M)"].min(),
                                          end=merged_data["Net Income (US $M)"].max())}

    # Add the income line renderer, using the secondary y-axis
    income_line = p.line(x="Quarter", y="Net Income (US $M)", source=source, line_width=2,
                         color="green", y_range_name="income", legend_label="Income")
    p.circle(x="Quarter", y="Net Income (US $M)", source=source,
             color="green", y_range_name="income")

    # Create and configure the secondary y-axis
    income_axis = LinearAxis(y_range_name="income", axis_label="Income (US $M)",
                             axis_line_color="green", major_label_text_color="green",
                             major_tick_line_color="green", minor_tick_line_color="green")
    p.add_layout(income_axis, 'right')

    # Create a HoverTool for the revenue line
    hover_revenue = HoverTool(renderers=[revenue_line], tooltips=[
        ("Date", "@Quarter{%F}"),
        ("Revenue (US $M)", "@{Revenue (US $M)}"),
        ("Income (US $M)", "@{Net Income (US $M)}")
    ],
                              formatters={
                                  '@Quarter': 'datetime',  # use 'datetime' formatter for 'Quarter' field
                                  '@{Revenue (US $M)}': 'printf',  # use 'printf' formatter for 'Revenue' field
                                  '@{Net Income (US $M)}': 'printf'
                              },
                              mode='vline'
                              )

    # Configure the legend to appear in the top-left corner
    p.legend.location = "top_left"
    p.legend.title = 'Legend'
    p.legend.title_text_font_size = '10pt'
    p.legend.label_text_font_size = '10pt'

    # Add the HoverTools to the figure
    p.add_tools(hover_revenue)

    # Get Chart Components
    script, div = components(p)

    return render_template(
        template_name_or_list='forecasting.html',
        script=script,
        div=div
    )


@app.route('/forecast_output', methods=['POST'])
def forecast():
    user_input = request.form.get('user_input')
    print(user_input)
    model_revenue = pickle.load(open("./models/revenue_model_arima.pkl", "rb"))
    model_income = pickle.load(open("./models/net_income_model_tes.pkl", "rb"))

    revenue_forecast = round(np.exp(model_revenue.forecast(int(user_input))), 2)
    income_forecast = model_income.forecast(int(user_input))

    revenue_data = pd.read_pickle("./models/revenue_data.pkl")
    income_data = pd.read_pickle("./models/income_data.pkl")

    # Reset the index to ensure 'Quarter' is a column
    revenue_data.reset_index(inplace=True)
    income_data.reset_index(inplace=True)

    df_revenue_forecast = revenue_forecast.reset_index()
    df_revenue_forecast.columns = ['Quarter', 'Revenue (US $M)']

    df_income_forecast = income_forecast.reset_index()
    df_income_forecast.columns = ['Quarter', 'Net Income (US $M)']

    df_revenue_final = pd.concat([revenue_data, df_revenue_forecast])
    df_income_final = pd.concat([income_data, df_income_forecast])

    # Merge the two dataframes on 'Quarter'
    merged_data = pd.merge(df_revenue_final, df_income_final, on='Quarter')

    # Create a ColumnDataSource from the merged data
    source = ColumnDataSource(data=merged_data)

    # Create a new plot with a datetime axis type
    p = figure(x_axis_type="datetime", title="Revenue and Income Over Time", height=470, width=1260)
    p.title.text_font_size = '15pt'
    p.title.text_font = 'cursive'
    p.title.text_align = 'right'

    # Add the revenue line renderer with legend and line thickness
    revenue_line = p.line(x="Quarter", y="Revenue (US $M)", source=source, line_width=2,
                          color="blue", legend_label="Revenue")
    p.circle(x="Quarter", y="Revenue (US $M)", source=source, color="blue")

    # Configure the primary y-axis
    p.yaxis.axis_label = "Revenue (US $M)"
    p.yaxis.axis_line_color = "blue"
    p.yaxis.major_label_text_color = "blue"
    p.yaxis.major_tick_line_color = "blue"
    p.yaxis.minor_tick_line_color = "green"

    # Add a secondary y-axis for income data
    p.extra_y_ranges = {"income": Range1d(start=merged_data["Net Income (US $M)"].min(),
                                          end=merged_data["Net Income (US $M)"].max())}

    # Add the income line renderer, using the secondary y-axis
    income_line = p.line(x="Quarter", y="Net Income (US $M)", source=source, line_width=2,
                         color="green", y_range_name="income", legend_label="Income")
    p.circle(x="Quarter", y="Net Income (US $M)", source=source,
             color="green", y_range_name="income")

    # Create and configure the secondary y-axis
    income_axis = LinearAxis(y_range_name="income", axis_label="Income (US $M)",
                             axis_line_color="green", major_label_text_color="green",
                             major_tick_line_color="green", minor_tick_line_color="green")
    p.add_layout(income_axis, 'right')

    # Create a HoverTool for the revenue line
    hover_revenue = HoverTool(renderers=[revenue_line], tooltips=[
        ("Date", "@Quarter{%F}"),
        ("Revenue (US $M)", "@{Revenue (US $M)}"),
        ("Income (US $M)", "@{Net Income (US $M)}")
    ],
                              formatters={
                                  '@Quarter': 'datetime',  # use 'datetime' formatter for 'Quarter' field
                                  '@{Revenue (US $M)}': 'printf',  # use 'printf' formatter for 'Revenue' field
                                  '@{Net Income (US $M)}': 'printf'
                              },
                              mode='vline'
                              )

    # Configure the legend to appear in the top-left corner
    p.legend.location = "top_left"
    p.legend.title = 'Legend'
    p.legend.title_text_font_size = '10pt'
    p.legend.label_text_font_size = '10pt'

    # Add the HoverTools to the figure
    p.add_tools(hover_revenue)

    # Get Chart Components
    script, div = components(p)

    return render_template(
        template_name_or_list='forecasting.html',
        script=script,
        div=div
    )

def sales_by_year():
    sales_year_df = pd.read_pickle("./models/sales_year_pivot.pkl")
    # Sample data
    data = {
        'Month': sales_year_df['Month'].to_list(),
        "2010": sales_year_df[2010].to_list(),
        "2011": sales_year_df[2011].to_list(),
        "2012": sales_year_df[2012].to_list()
    }

    # Convert data to a DataFrame
    df = pd.DataFrame(data)

    # Convert month to a datetime type to plot correctly
    df['Month'] = pd.to_datetime(df['Month'], format='%B')
    df['Month_Num'] = df['Month'].dt.strftime('%m')
    df['Month'] = df['Month'].dt.strftime('%b')  # Keep only the month name for display
    df.sort_values('Month_Num', ascending=True, inplace=True)

    # Create a ColumnDataSource from the DataFrame
    source = ColumnDataSource(df)

    # Create a new plot with a categorical x-axis for the months
    fig = figure(x_range=df['Month'], title="Monthly Sales Over Years", height=400, width=700)
    fig.title.text_font_size = '15pt'
    fig.xaxis.axis_label = 'Month'
    fig.yaxis.axis_label = 'Sales'

    # Add line renderers for each year's sales
    colors = ["blue", "green", "red"]
    years = ["2010", "2011", "2012"]
    for year, color in zip(years, colors):
        fig.line(x='Month', y=year, source=source, line_width=2, color=color, legend_label=year)
        fig.circle(x='Month', y=year, source=source, size=5, color=color, legend_label=year)

    # Create a HoverTool
    hover = HoverTool(tooltips=[
        ("Month", "@Month"),
        ("Sales", "$y{0,0}")
    ])

    # Add the HoverTool to the figure
    fig.add_tools(hover)

    # Get Chart Components
    script_sales, div_sales = components(fig)

    return script_sales,div_sales


@app.route('/sales_prediction')
def sales_prediction():
    script_sales, div_sales = sales_by_year()

    return render_template(
        'sales_prediction.html',
        script=script_sales,
        div=div_sales
    )


@app.route('/sales_prediction_output', methods=['POST'])
def sales_prediction_output():
    model_sales = pickle.load(open("./models/sales_prediction_model.pkl", "rb"))
    # Get Chart Components
    script_sales, div_sales = sales_by_year()

    if request.method == "POST":
        holiday_flag_inp = request.form['holiday_flag']
        if holiday_flag_inp == "holiday_yes":
            Holiday_Flag = 1
        else:
            Holiday_Flag = 0

        Temperature = float(request.form['temperature'])

        CPI = float(request.form['cpi'])

        Unemployment = float(request.form['Unemployment'])

        Store = int(request.form['store_id'])

        values_sales = np.array([Store, Holiday_Flag, Temperature, CPI, Unemployment])

        prediction_sales = model_sales.predict(values_sales.reshape(1, 5))
        prediction_sales = round(prediction_sales[0],2)

        return render_template(
            'sales_prediction.html',
            script=script_sales,
            div=div_sales,
            sales_prediction=prediction_sales
        )


@app.route('/churn_prediction')
def churn_prediction():
    return render_template('churn_prediction.html')


@app.route('/churn_prediction_output', methods=['POST'])
def churn_prediction_output():
    model_churn = pickle.load(open("./models/churn_prediction_model.pkl", "rb"))

    if request.method == "POST":
        gender_inp = request.form['gender']
        if gender_inp == "gender_male":
            gender = 1
        else:
            gender = 0

        tenure = int(request.form['tenure'])

        seniorcitizen_inp = request.form['Seniorcitizen']
        if seniorcitizen_inp == "seniorcitizen_yes":
            SeniorCitizen = 1
        else:
            SeniorCitizen = 0

        dependent_inp = request.form['Dependent']
        if dependent_inp == "dependent_yes":
            Dependents = 1
        else:
            Dependents = 0

        techsupport_inp = request.form['techsupport']
        if techsupport_inp == "techsupport_yes":
            TechSupport_Yes = 1
        else:
            TechSupport_Yes = 0

        partner_inp = request.form['partner']
        if partner_inp == "partner_yes":
            Partner = 1
        else:
            Partner = 0

        MonthlyCharges = float(request.form['monthlycharges'])

        phoneservice_inp = request.form['phoneservice']
        if phoneservice_inp == "phoneservice_yes":
            PhoneService = 1
        else:
            PhoneService = 0

        onlinesecurity_inp = request.form['onlinesecurity']
        if onlinesecurity_inp == "onlinesecurity_yes":
            OnlineSecurity_Yes = 1
        else:
            OnlineSecurity_Yes = 0

        streamingtv_inp = request.form['streamingtv']
        if streamingtv_inp == "streamingtv_yes":
            StreamingTV_Yes = 1
        else:
            StreamingTV_Yes = 0

        Multiplelines_inp = request.form['Multiplelines']
        if Multiplelines_inp == "multiplelines_yes":
            MultipleLines_Yes = 1
            MultipleLines_No_phone_service = 0
        elif Multiplelines_inp == "multiplelines_no_phoneservice":
            MultipleLines_Yes = 0
            MultipleLines_No_phone_service = 1
        else:
            MultipleLines_Yes = 0
            MultipleLines_No_phone_service = 0

        TotalCharges = float(request.form['totalcharges'])

        internetservice_inp = request.form['internetservice']
        if internetservice_inp == "internetservice_fibreoptic":
            InternetService_Fiber_optic = 1
        else:
            InternetService_Fiber_optic = 0

        onlinebackup_inp = request.form['onlinebackup']
        if onlinebackup_inp == "onlinebackup_yes":
            OnlineBackup_Yes = 1
        else:
            OnlineBackup_Yes = 0

        streamingmovies_inp = request.form['streamingmovies']
        if streamingmovies_inp == "streamingmovies_yes":
            StreamingMovies_Yes = 1
        else:
            StreamingMovies_Yes = 0

        Deviceprotection_inp = request.form['Deviceprotection']
        if Deviceprotection_inp == "deviceprotection_yes":
            DeviceProtection_Yes = 1
        else:
            DeviceProtection_Yes = 0

        Paymentmethod_inp = request.form['Paymentmethod']
        if Paymentmethod_inp == "paymentmethod_echeck":
            PaymentMethod_Credit_card = 0
            PaymentMethod_Mailed_check = 0
            PaymentMethod_Electronic_check = 1
        elif Paymentmethod_inp == "paymentmethod_mailedcheck":
            PaymentMethod_Credit_card = 0
            PaymentMethod_Mailed_check = 1
            PaymentMethod_Electronic_check = 0
        elif Paymentmethod_inp == "paymentmethod_creditcard":
            PaymentMethod_Credit_card = 1
            PaymentMethod_Mailed_check = 0
            PaymentMethod_Electronic_check = 0
        else:
            PaymentMethod_Credit_card = 0
            PaymentMethod_Mailed_check = 0
            PaymentMethod_Electronic_check = 0

        contract_inp = request.form['contract']
        if contract_inp == "contract_oneyear":
            Contract_One_year = 1
            Contract_Two_year = 0
        elif contract_inp == "contract_twoyear":
            Contract_One_year = 0
            Contract_Two_year = 1
        else:
            Contract_One_year = 0
            Contract_Two_year = 0

        paperless_billing_inp = request.form['paperless_billing']
        if paperless_billing_inp == "paperless_billing_yes":
            PaperlessBilling = 1
        else:
            PaperlessBilling = 0

        values = np.array([SeniorCitizen,Partner,Dependents,tenure,PhoneService,PaperlessBilling,
                           MonthlyCharges,TotalCharges,MultipleLines_No_phone_service,MultipleLines_Yes,
                           InternetService_Fiber_optic, OnlineSecurity_Yes, OnlineBackup_Yes, DeviceProtection_Yes,
                           TechSupport_Yes, StreamingTV_Yes, StreamingMovies_Yes, Contract_One_year, Contract_Two_year,
                           PaymentMethod_Credit_card, PaymentMethod_Electronic_check, PaymentMethod_Mailed_check])

        prediction = model_churn.predict(values.reshape(1, 22))
        prediction = prediction[0]

        if prediction == 1:
            prediction_output = "Yes"
        else:
            prediction_output = "No"

        return render_template(
            'churn_prediction.html',
            churn_prediction=prediction_output)


@app.route('/cust_segmentation')
def cust_segmentation():
    return render_template('cust_segmentation.html')


@app.route('/cust_segmentation_output', methods=['POST'])
def cust_segmentation_output():
    rfm_data = pickle.load(open("./models/rfm_model.pkl", "rb"))
    df_segment = pd.DataFrame()
    data_dict = request.form.to_dict()
    input_high = ""
    input_avg = ""
    input_low = ""

    if "HighPriorityCustomer" in data_dict.keys():
        input_high = data_dict['HighPriorityCustomer']

    if "AveragePriorityCustomer" in data_dict.keys():
        input_avg = data_dict['AveragePriorityCustomer']

    if "LowPriorityCustomer" in data_dict.keys():
        input_low = data_dict['LowPriorityCustomer']

    if input_high == "on":
        segment_high = "High-Priority Customer"
        recency_high = round(rfm_data[rfm_data['Cust_Segment'] == segment_high]['Recency'].mean(), 0)
        freq_high = round(rfm_data[rfm_data['Cust_Segment'] == segment_high]['Frequency'].mean(), 0)
        monetary_high = round(rfm_data[rfm_data['Cust_Segment'] == segment_high]['Monetory'].mean(), 2)
        df_high = rfm_data[rfm_data['Cust_Segment'] == segment_high][
            ['CustomerID', 'Recency', 'Frequency', 'Monetory', 'Cust_Segment']]
        df_segment = df_segment.append(df_high)
    else:
        recency_high = ""
        freq_high = ""
        monetary_high = ""

    if input_avg == "on":
        segment_avg = "Mid-Priority Customer"
        recency_avg = round(rfm_data[rfm_data['Cust_Segment'] == segment_avg]['Recency'].mean(), 0)
        freq_avg = round(rfm_data[rfm_data['Cust_Segment'] == segment_avg]['Frequency'].mean(), 0)
        monetary_avg = round(rfm_data[rfm_data['Cust_Segment'] == segment_avg]['Monetory'].mean(), 2)
        df_avg = rfm_data[rfm_data['Cust_Segment'] == segment_avg][
            ['CustomerID', 'Recency', 'Frequency', 'Monetory', 'Cust_Segment']]
        df_segment = df_segment.append(df_avg)
    else:
        recency_avg = ""
        freq_avg = ""
        monetary_avg = ""

    if input_low == "on":
        segment_low = "Low-Priority Customer"
        recency_low = round(rfm_data[rfm_data['Cust_Segment'] == segment_low]['Recency'].mean())
        freq_low = round(rfm_data[rfm_data['Cust_Segment'] == segment_low]['Frequency'].mean(), 0)
        monetary_low = round(rfm_data[rfm_data['Cust_Segment'] == segment_low]['Monetory'].mean(), 2)
        df_low = rfm_data[rfm_data['Cust_Segment'] == segment_low][
            ['CustomerID', 'Recency', 'Frequency', 'Monetory', 'Cust_Segment']]
        df_segment = df_segment.append(df_low)
    else:
        recency_low = ""
        freq_low = ""
        monetary_low = ""

    if df_segment.shape[0] != 0:
        df_segment = df_segment[df_segment['Monetory'] > 0]
        df_segment['Monetory'] = df_segment['Monetory'].apply(lambda x: round(x, 2))

    return render_template(
        template_name_or_list='cust_segmentation.html',
        high_recency_val=recency_high,
        high_freq_val=freq_high,
        high_monetary_val=monetary_high,
        avg_recency_val=recency_avg,
        avg_freq_val=freq_avg,
        avg_monetary_val=monetary_avg,
        low_recency_val=recency_low,
        low_freq_val=freq_low,
        low_monetary_val=monetary_low,
        dataframe=df_segment.values
    )


if __name__ == "__main__":
    app.run(debug=False)
