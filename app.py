import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import datetime


st.set_page_config(page_title="Learning Beautiful Dashboard",
                   page_icon=":tada:", layout="wide", initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Functions to call the pages

def sales_dashboard(sales_excel):
    st.header("Sales Dashboard")
    # Row A
    sales_df = pd.read_excel(sales_excel, sheet_name='Sales')
    col1, col2 = st.columns((5, 5))
    with col1:  # If interactive, these will change
        fig = go.Figure(data=[go.Pie(labels=sales_df['Product Topic'],
                                     values=sales_df['order_amount'])])
        fig.update_traces(hole=.4, hoverinfo="label+percent+value")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # If interactive, these will change
        customer_counts = sales_df['customer_type'].value_counts()
        fig = go.Figure(
            data=[go.Bar(x=customer_counts.index, y=customer_counts)])
        fig.update_layout(title="Number of Customers by Type",
                          xaxis_title="Customer Type",
                          yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    # Row B
    c1, c2 = st.columns((5, 5))
    with c1:
        years = [2021, 2022, 2023]
        selected_years = st.multiselect("Select years to view", years)
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
    sales_df['year'] = sales_df['order_date'].dt.year
    sales_df['month'] = sales_df['order_date'].dt.month
    monthly_sales = sales_df.groupby(['year', 'month'])[
        'order_amount'].sum().reset_index()
    # filter the monthly_sales dataframe based on the years selected by the user
    monthly_sales = monthly_sales[monthly_sales['year'].isin(selected_years)]
    monthly_sales['month_name'] = monthly_sales['month'].apply(
        lambda x: datetime.datetime.strptime(str(x), "%m").strftime("%B"))
    fig = go.Figure(data=[go.Scatter(x=monthly_sales['month_name'], y=monthly_sales['order_amount'],
                    mode='lines', name=str(year)) for year in monthly_sales['year'].unique()])
    fig.update_layout(title="Yearly Sales by Month",
                      xaxis_title="Month",
                      yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)

    with c2:
        pass
    # Row C
    column_name = "customer_type"  # If interactive, these will change
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
    sales_by_customer_type = sales_df.groupby(['year', column_name])[
        'order_amount'].sum().reset_index()
    sales_by_customer_type_pivot = sales_by_customer_type.pivot(
        index='year', columns=column_name, values='order_amount').reset_index()
    cols = sales_by_customer_type_pivot.columns
    data = []
    for col in cols:
        data.append(go.Bar(
            name=col, x=sales_by_customer_type_pivot['year'], y=sales_by_customer_type_pivot[col]))
    fig = go.Figure(data=data)
    fig.update_layout(title=f"Sales Amount per {column_name} over the Years",
                      xaxis_title="Year",
                      yaxis_title="Sales Amount")
    fig.update_layout(barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

    # Row D
    countries_df = pd.read_excel(sales_excel, sheet_name='countries')
    sales_by_country = sales_df.groupby('country').agg(
        {'order_amount': 'sum'}).reset_index()
    merged_data = pd.merge(countries_df, sales_by_country, on='country')

    fig = go.Figure()

    # Create trace for the map data
    fig.add_trace(go.Scattermapbox(
        lat=merged_data['latitude'],
        lon=merged_data['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=merged_data['order_amount']/1000,
            color=merged_data['order_amount'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Sales Amount',
                thickness=20,
                ticklen=3)),
        text=merged_data['country']
    ))

    # Set mapbox style and center
    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            # replace with your own Mapbox token
            accesstoken="pk.eyJ1IjoiYnRqajE5OTkiLCJhIjoiY2t1YXd5YXA4MGw5bjJwcTY1bGx2bW96eiJ9.XMqwz51XPwCSktVr-qxnWg",
            style='mapbox://styles/mapbox/streets-v11',
            center=dict(lon=merged_data['longitude'].median(
            ), lat=merged_data['latitude'].median()),
            zoom=6)
    )

    # Set chart title
    fig.update_layout(title='Total Sales by Country')

    # Display the map in Streamlit
    st.plotly_chart(fig, use_container_width=True, height=1000)


def cx_dashboard(cx_dataframe):
    st.header("Customer Experience Dashboard")
    # Row A
    col1, col2 = st.columns((5, 5))
    with col1:
        churn_df = pd.read_excel(cx_dataframe, sheet_name='Churn')
        toys_df = pd.read_excel(cx_dataframe, sheet_name='Toys')
        # st.table(churn_df)
        # st.table(toys_df)
        fig_toys = go.Figure(data=[go.Pie(labels=toys_df['Toy Category'])])
        fig_toys.update_traces(hole=.4, hoverinfo="label+percent+value")
        fig_toys.update_layout(title="Toy Category")
        st.plotly_chart(fig_toys, use_container_width=True)

    with col2:
        grouped_data = toys_df.groupby('Toy Category')[
            'Satisfaction Level'].mean().reset_index()

        fig = px.bar(grouped_data, x='Toy Category',
                     y='Satisfaction Level', color='Toy Category')
        fig.update_layout(title='Average Satisfaction Level by Toy Category',
                          xaxis_title='Toy Category', yaxis_title='Average Satisfaction Level')

        st.plotly_chart(fig)

    # Row B
    col1, col2 = st.columns((5, 5))
    with col1:
        fig = px.histogram(toys_df, x='Average Total Question Score', nbins=30, color_discrete_sequence=[
                           '#636EFA'], hover_data=['Toy Category', 'Toy Name'])
        fig.update_layout(title='Distribution of Average Total Question Scores',
                          xaxis_title='Quiz Score', yaxis_title='Frequency')
        B_col1_color = st.color_picker(
            "Select a Color", '#636EFA', key="B_col1_color")
        fig.update_traces(marker=dict(color=B_col1_color))
        st.plotly_chart(fig)

    with col2:
        fig = px.histogram(toys_df, x='Satisfaction Level', nbins=30, color_discrete_sequence=[
                           "#636EFA"], hover_data=['Toy Category', 'Toy Name'])
        fig.update_layout(title='Distribution of Satisfaction Levels',
                          xaxis_title='Satisfaction Level', yaxis_title='Frequency')
        B_col2_color = st.color_picker(
            "Select a Color", '#636EFA', key="B_col2_color")
        fig.update_traces(marker=dict(color=B_col2_color))
        st.plotly_chart(fig)

    # Row C
    col1, col2 = st.columns((5, 5))
    with col1:
        st.header("Predicting Churn")
        # Beatrice
        st.subheader("The Top 3 features affecting customer Churn Rate:")
        # create a DataFrame with the feature names
        features_df = pd.DataFrame({
            'Rank': [1, 2, 3],
            'Feature': ['Last Website Visit', 'Total Transactions Made', 'Time Spent on App in Minutes(Yearly)']
        }).set_index('Rank')
        st.table(features_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'left')]}]))
        # st.table(df.style.set_table_styles([{'selector': 'table', 'props': [('width', '50%')]}]))
        # Classification Report
        st.markdown("Testing accuracy: 0.985")
        # precision recall f1 support
        data = {'precision': [0.98, 0.00],
                'recall': [1.00, 0.00],
                'f1-score': [0.99, 0.00],
                'support': [197, 3]}
        index = ['False', 'True']
        df = pd.DataFrame(data, index=index)
        st.table(df)
        # accuracy macro avg weighted avg
        data = {
            'metric': ['accuracy', 'macro avg', 'weighted avg'],
            'precision': ['-', '0.49', '0.97'],
            'recall': ['-', '0.50', '0.98'],
            'f1-score': ['0.98', '0.50', '0.98'],
            'support': ['200', '200', '200']}
        df = pd.DataFrame(data).set_index('metric')
        st.table(df)
    with col2:
        pass

    # View Code Block
    with st.expander("View Code"):
        with open("churn_logistic_regression.py") as f:
            code = f.read()
        st.code(code, language="python")


### Landing Page #######################################################################################
st.title("Learning Beautiful Dashboard")

### Sidebar Items ######################################################################################
st.sidebar.title("Navigation")
uploaded_file_Sales = st.sidebar.file_uploader(
    "Upload Sales Transaction Dataset here", key="sales")
uploaded_file_CX = st.sidebar.file_uploader(
    "Upload Customer Dataset here", key="cx")

st.sidebar.subheader('Dashboards')
options = st.sidebar.radio(
    'Dashboard', options=["Sales Dashboard", "Customer Dashboard"])

st.sidebar.markdown('''
---
Created with ❤️ by Team 2
''')

# To call the pages created
if uploaded_file_Sales:
    sales_excel = uploaded_file_Sales
    if options == "Sales Dashboard":
        sales_dashboard(sales_excel)

if uploaded_file_CX:
    cx_excel = uploaded_file_CX
    if options == "Customer Dashboard":
        cx_dashboard(cx_excel)
