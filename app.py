import tempfile
import io
import fig as my_fig
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import warnings
import plotly.io as pio
import base64
from io import BytesIO
import seaborn as sns
from PIL import Image
import re
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Summary Statistics", page_icon=":bar_chart:", layout="wide")
st.title("VISA INVOICES")
image = Image.open('payip.png')

uploaded_file = st.file_uploader("Upload a CSV, XLSX, or TXT File", type=["csv", "xlsx", "txt"])

df = None  # Initialize df as None

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == '.xlsx':
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    elif file_extension == '.txt':
        with uploaded_file:
            df = pd.DataFrame({'text': uploaded_file.readlines()})

    # Ensure all datetime columns are converted to date
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].dt.date

    # Ensure no commas are present in the dataframe
    df = df.replace({',': ''}, regex=True)

    # Interactive Filters on Sidebar
    st.sidebar.header('Filters')
    filter_columns = [
        'Rate Type', 'Type', 'Billing Line',
        'Bin Map', 'Billing Currency', 'Description'
    ]
    filters = {}
    for col in filter_columns:
        if col in df.columns:
            options = list(df[col].unique())
            options.insert(0, 'All')
            selected = st.sidebar.selectbox(f"Select {col}", options)
            if selected != 'All':
                filters[col] = selected

    # Apply filters to dataframe
    for key, value in filters.items():
        df = df[df[key] == value]

    if 'Billing Date' in df.columns:
        df_filtered = df.dropna(subset=['Billing Date'])

        if pd.api.types.is_datetime64_any_dtype(df_filtered['Billing Date']):
            min_value = pd.to_datetime(df_filtered['Billing Date'].min())
            max_value = pd.to_datetime(df_filtered['Billing Date'].max())

            selected_min, selected_max = st.date_input("Select a date range for 'Billing Date'", [min_value, max_value])

            df = df[(df['Billing Date'] >= selected_min) & (df['Billing Date'] <= selected_max)]
        else:
            st.warning("'Billing Date' is not a recognized date format. Unable to create a slider.")

# Display the filtered df if it exists
if df is not None:
    col1, col2 = st.columns((2))
    df["Billing Period"] = pd.to_datetime(df["Billing Period"])
    startDate = pd.to_datetime(df["Billing Period"]).min()
    endDate = pd.to_datetime(df["Billing Period"]).max()

    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df["Billing Period"] >= date1) & (df["Billing Period"] <= date2)].copy()

    # Display the filtered df
    st.write(df)




    # Assuming df is your DataFrame
    df['Taxable Amount (Tax Currency)'] = pd.to_numeric(df['Taxable Amount (Tax Currency)'], errors='coerce')
    # Compute top analytics
    total_charge = int(df['Total'].sum())
    total_refunds = int(df['Refunds'].sum())
    total_tax = float(df['Taxable Amount (Tax Currency)'].sum())
    refunds_percentage = -((total_refunds / total_charge) * 100)
    billing_lines = df['Billing Line'].nunique()
    bin_maps = df['BIN Map'].nunique()
    total_rows = df.shape[0]
    largest_10_charges = df.nlargest(10, 'Total')
    sum_of_largest_10_charges = largest_10_charges['Total'].sum()
    percentage_largest_10_charges = (sum_of_largest_10_charges / total_charge) * 100

    total1, total2, total3, total4 = st.columns(4, gap='large')
    with total1:
        st.info('Total Charge', icon="📌")
        st.metric(label="Sum Total in $", value=f"{total_charge:,.0f}")

    with total2:
        st.info('Total Refunds', icon="📌")
        st.metric(label="Sum Refunds in $", value=f"{total_refunds:,.0f}")

    with total3:
        st.info('Total Taxable Amount', icon="📌")
        st.metric(label="Sum of Taxable Amount in ZAR", value=f"{total_tax:,.0f}")

    with total4:
        st.info('Refunds % of Total', icon="📌")
        st.metric(label="Refunds % of Total", value=f"{refunds_percentage :,.2f}%")

    total5, total6, total7, total8 = st.columns(4, gap='large')
    with total5:
        st.info('Number of Unique Billing Lines', icon="📌")
        st.metric(label="Unique Billing Lines Count", value=f"{billing_lines:,.0f}")

    with total6:
        st.info('Number of Unique Bin Maps', icon="📌")
        st.metric(label="Unique BiN Maps Count", value=f"{bin_maps:,.0f}")

    with total7:
        st.info('Number of Transactions (Line Items)', icon="📌")
        st.metric(label="Number of Line Items", value=f"{total_rows:,.0f}")

    with total8:
        st.info('Top 10 Charges % out of Total', icon="📌")
        st.metric(label="% of Top 10 Charges", value=f"{percentage_largest_10_charges:,.2f}%")

    # Calculate percentage of each Total Charge out of the sum
    largest_10_charges['Percentage'] = (largest_10_charges['Total'] / largest_10_charges['Total'].sum()) * 100

    # Sort by 'Total' column in descending order and then take the top 10
    largest_10_charges_sorted = largest_10_charges.sort_values(by='Total', ascending=False).head(10)

    # Create a bar graph with each "Billing Line" as its own series
    fig = go.Figure()

    for _, row in largest_10_charges_sorted.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row['Total']],
                y=[row['Billing Line']],
                name=row['Billing Line'],
                orientation='h',
                hoverinfo='x+name+text',  # Include the customized hover text
                hovertemplate=f"Total: {row['Total']}<br>Billing Line: {row['Billing Line']}<br>Description: {row.get('Description', '')}",
                # Custom hover template
                text=f"{row['Total']}<br>{row['Percentage']:.2f}%",
                # Displaying both the actual value and percentage on the bars
                textposition='outside'
            )
        )

    fig.update_layout(
        title='Top 10 Billing Lines by Total',
        xaxis_title="Total",
        yaxis_title="Billing Line",
        yaxis_categoryorder='total ascending',
        barmode='stack'
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    # Bar graph for 'Type' vs. 'Total'
    type_totals = df.groupby('Type')['Total'].sum().reset_index()
    fig_type_bar = px.bar(type_totals, x='Type', y='Total', title='Total by Type', color='Type')

    # Calculate the percentage of total for each type
    type_totals['Percentage'] = (type_totals['Total'] / type_totals['Total'].sum()) * 100

    # Add annotations for the percentage
    annotations = []
    for i, row in type_totals.iterrows():
        annotations.append(
            dict(
                x=row['Type'],
                y=row['Total'] + 0.02 * type_totals['Total'].max(),  # position the annotation slightly above the bar
                text=f"{row['Percentage']:.2f}%",
                showarrow=False,
                font=dict(size=10)
            )
        )
    fig_type_bar.update_layout(annotations=annotations)
    st.plotly_chart(fig_type_bar)
    # Average of Rate grouped by Billing Period
    if 'Rate' in df.columns and 'Billing Period' in df.columns:
        avg_rate_by_period = df.groupby('Billing Period')['Rate'].mean().reset_index()

        # Create the line graph using Plotly Express
        fig_rate_avg = px.line(avg_rate_by_period, x='Billing Period', y='Rate', title='Billing Period by Average Rate')
        st.plotly_chart(fig_rate_avg)

    if 'Total' in df.columns and 'Billing Period' in df.columns:
        total_by_period = df.groupby('Billing Period')['Total'].sum().reset_index()

        # Create the line graph using Plotly Express
        fig_ratettl = px.line(total_by_period, x='Billing Period', y='Total', title='Billing Period by Total Charge')
        st.plotly_chart(fig_ratettl)

    if 'Units' in df.columns and 'Billing Period' in df.columns:
        units_by_period = df.groupby('Billing Period')['Units'].sum().reset_index()

        # Create the line graph using Plotly Express
        fig_rateut = px.line(units_by_period, x='Billing Period', y='Units', title='Billing Period by Total Units')
        st.plotly_chart(fig_rateut)

    if 'Refunds' in df.columns and 'Billing Period' in df.columns:
        ref_byperiod = df.groupby('Billing Period')['Refunds'].sum().reset_index()

        # Create the line graph using Plotly Express
        fig_rf = px.line(ref_byperiod, x='Billing Period', y='Refunds', title='Billing Period by Total Refund ')
        st.plotly_chart(fig_rf)

    st.markdown("""---""")


    def plot_to_html(fig):
        '''Converts a plotly figure to an HTML string'''
        return pio.to_html(fig, full_html=False)


    def metric_to_html(label, value, icon=""):
        """Converts metric to HTML with inline styles"""
        return f"""
        <div style="border: 1px solid #E5E5E5; border-radius: 4px; padding: 20px; background-color: #f9f9f9;">
            <h4 style="color: #2c3e50;">{icon} {label}</h4>
            <h2 style="color: #e74c3c;">{value}</h2>
        </div>
        """


    # Generate HTML for metrics
    metrics_html = ""
    metrics_data = [
        ("Sum Total in $", f"{total_charge:,.0f}", "📌"),
        ("Sum Refunds in $", f"{total_refunds:,.0f}", "📌"),
        ("Sum of Taxable Amount in ZAR", f"{total_tax:,.0f}", "📌"),
        ("Refunds % of Total", f"{refunds_percentage :,.2f}%", "📌"),
        ("Unique Billing Lines Count", f"{billing_lines:,.0f}", "📌"),
        ("Unique BiN Maps Count", f"{bin_maps:,.0f}", "📌"),
        ("Number of Line Items", f"{total_rows:,.0f}", "📌"),
        ("% of Top 10 Charges", f"{percentage_largest_10_charges:,.2f}%", "📌"),
    ]
    for label, value, icon in metrics_data:
        metrics_html += metric_to_html(label, value, icon)

    # Combine metrics HTML with charts' HTML
    all_figs_html = metrics_html
    for fig in [fig_type_bar, fig_rate_avg, fig_ratettl, fig_rateut, fig_rf]:  # Corrected the figures list here
        all_figs_html += plot_to_html(fig)

    # Provide a download link for the combined HTML
    b64 = base64.b64encode(all_figs_html.encode()).decode()  # Encoding the HTML string to bytes and then to Base64
    href = f'<a href="data:text/html;base64,{b64}" download="dashboard.html">Download the Dashboard as HTML</a>'
    st.markdown(href, unsafe_allow_html=True)


    def get_table_download_link(df, filename="data.xlsx", link_name="Download Table"):
        """Generates a link allowing the data in a given pandas dataframe to be downloaded as Excel"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        xlsx_data = output.getvalue()
        b64 = base64.b64encode(xlsx_data).decode()  # B64 encoding
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_name}</a>'
        return href


    if df is not None:
        # Your existing code here...

        # Find unique Billing Lines
        unique_billing_lines = df['Billing Line'].unique()

        # Initializing a list to store data for each billing line
        billing_data = []

        for line in unique_billing_lines:
            # Filter df for the current Billing Line
            temp_df = df[df['Billing Line'] == line]

            # Determine the Latest and Last Billing Dates for the current Billing Line
            latest_billing_date = temp_df["Billing Period"].max()
            last_billing_date = temp_df[temp_df["Billing Period"] < latest_billing_date]["Billing Period"].min()

            # Capture unique descriptions for the current Billing Line
            unique_descriptions = temp_df['Description'].unique()
            concatenated_description = ", ".join(unique_descriptions)

            # Checking if the last_billing_date exists (i.e., there's at least two dates for the Billing Line)
            if pd.notna(last_billing_date):
                billing_data.append([last_billing_date, latest_billing_date, line, concatenated_description])

        # Create a dataframe from the billing_data list
        billing_matrix = pd.DataFrame(billing_data,
                                      columns=['Oldest_billing_date', 'Latest_billing_date', 'Billing Line',
                                               'Description'])
        # Convert datetime columns to date only (without time component)
        billing_matrix['Oldest_billing_date'] = billing_matrix['Oldest_billing_date'].dt.date
        billing_matrix['Latest_billing_date'] = billing_matrix['Latest_billing_date'].dt.date
        # Display the matrix/table
        st.table(billing_matrix)

        # Display the download link for the matrix
        st.markdown(get_table_download_link(billing_matrix), unsafe_allow_html=True)


    def get_table_download_link(df, filename="data.xlsx", link_name="Download Table"):
        """Generates a link allowing the data in a given pandas dataframe to be downloaded as Excel"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        xlsx_data = output.getvalue()
        b64 = base64.b64encode(xlsx_data).decode()  # B64 encoding
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_name}</a>'
        return href


    if df is not None:

        # Find unique Rate Type
        unique_ratetype = df['Rate Type'].unique()

        # Initializing a list to store data for each rate type
        rate_type = []

        for type in unique_ratetype:
            # Filter df for the current Rate Type
            temp_df = df[df['Rate Type'] == type]

            # Determine the Latest and Last Billing Dates for the current Rate Type
            latest_billing_date = temp_df["Billing Period"].max()
            last_billing_date = temp_df[temp_df["Billing Period"] < latest_billing_date]["Billing Period"].min()

            # Checking if the last_billing_date exists (i.e., there's at least two dates for the Rate Type)
            if pd.notna(last_billing_date):
                rate_type.append([last_billing_date, latest_billing_date, type])

        # Create a dataframe from the billing_data list
        billing_matrix = pd.DataFrame(rate_type,
                                      columns=['Oldest_billing_date', 'Latest_billing_date', 'Rate Type'])

        # Convert datetime columns to date only (without time component)
        billing_matrix['Oldest_billing_date'] = billing_matrix['Oldest_billing_date'].dt.date
        billing_matrix['Latest_billing_date'] = billing_matrix['Latest_billing_date'].dt.date

        # Display the matrix/table
        st.table(billing_matrix)
        # Display the download link for the matrix
        st.markdown(get_table_download_link(billing_matrix), unsafe_allow_html=True)

        # ... [All your existing code remains unchanged]

        # For displaying the new billing lines on their first date of appearance
        new_billing_data = []

        for line in unique_billing_lines:
            # Filter df for the current Billing Line
            temp_df = df[df['Billing Line'] == line]

            # Determine the First Billing Date for the current Billing Line
            first_billing_date = temp_df["Billing Period"].min()

            # Extracting the row corresponding to the first billing date
            first_date_row = temp_df[temp_df["Billing Period"] == first_billing_date].iloc[0]

            new_billing_data.append(
                [line, first_date_row["Billing Period"], first_date_row["Description"], first_date_row["Total"]])

        # Create a dataframe from the new_billing_data list
        new_billing_matrix = pd.DataFrame(new_billing_data,
                                          columns=['Billing Line', 'Billing Date of First Appearance', 'Description',
                                                   'Total'])
        # Convert datetime column to date only (without time component)
        new_billing_matrix['Billing Date of First Appearance'] = new_billing_matrix[
            'Billing Date of First Appearance'].dt.date
        # Display the matrix/table
        st.table(new_billing_matrix)

        # Display the download link for the matrix
        st.markdown(get_table_download_link(new_billing_matrix, filename="new_billing_lines.xlsx",
                                            link_name="Download New Billing Lines Table"), unsafe_allow_html=True)
    import base64
    import pandas as pd
    import streamlit as st
    from io import BytesIO


    def highlight_changes(s):
        is_highlight = ((s.pct_change().abs() > 0.1) & (s.index != s.index[0])) | (
                (s.shift(1).isna()) & (~s.isna()) & (s.index != s.index[0]))
        return ['background-color: yellow' if v else '' for v in is_highlight]


    def make_download_link(styled_pivot, pivot_name):
        output = BytesIO()
        styled_pivot.to_excel(output, engine='openpyxl', index=True)
        xlsx_data = output.getvalue()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(xlsx_data).decode()}" download="{pivot_name}_pivot_table.xlsx">Download {pivot_name} pivot table as XLSX</a>'


    if df is not None and all(col in df.columns for col in
                              ['Rate', 'Billing Period', 'Description', 'Billing Line', 'Units', 'Total', 'Refunds']):
        df['Billing Period'] = df['Billing Period'].dt.date

        pivot_types = {
            'Rate': 'mean',
            'Units': 'sum',
            'Total': 'sum',
            'Refunds': 'sum'
        }

        for pivot_name, agg_func in pivot_types.items():
            pivot_df = pd.pivot_table(df, values=pivot_name, index=['Description', 'Billing Line'],
                                      columns='Billing Period', aggfunc=agg_func, margins=True,
                                      margins_name="Grand Total")

            level_option = st.selectbox(f'Select Index Level for {pivot_name}',
                                        ['Description', 'Description & Billing Line'])
            if level_option == 'Description':
                displayed_pivot = pivot_df.groupby('Description').sum()
            else:
                displayed_pivot = pivot_df

            sort_order = st.selectbox(f'Sort Grand Total by for {pivot_name}', ['Ascending', 'Descending'])
            if sort_order == 'Ascending':
                displayed_pivot = displayed_pivot.sort_values(by='Grand Total')
            else:
                displayed_pivot = displayed_pivot.sort_values(by='Grand Total', ascending=False)

            styled_pivot = displayed_pivot.style.apply(highlight_changes, axis=1, subset=displayed_pivot.columns[:-1])
            st.header(f"Pivot Table for {pivot_name}")  # Title for the pivot table
            st.dataframe(styled_pivot)

            download_link = make_download_link(styled_pivot, pivot_name)
            st.markdown(download_link, unsafe_allow_html=True)












