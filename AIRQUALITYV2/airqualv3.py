# -*- coding: utf-8 -*-
"""AIRQUALV3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10qEUbq0iAyh15F_qhToKjsS1rQJR3GuE

# Proyek Analisis Data: [Air Quality District in Beijing, China]
- **Nama:** [Shafida Afifah Firdausy]
- **Email:** [m279d4kx3233@bangkit.academy]
- **ID Dicoding:** [shafiidaaaa]
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import streamlit as st
import datetime
import matplotlib.cm as cm



aotizhongxin_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv")

changping_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Changping_20130301-20170228.csv")

dingling_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv")

dongsi_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv")

wanshouxigong_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Wanshouxigong_20130301-20170228.csv")

wanliu_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Wanliu_20130301-20170228.csv")

tiantan_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Tiantan_20130301-20170228.csv")

shunyi_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Shunyi_20130301-20170228.csv")

nongzhanguan_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Nongzhanguan_20130301-20170228.csv")

huairou_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Huairou_20130301-20170228.csv")

gucheng_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Gucheng_20130301-20170228.csv")

guanyuan_df = pd.read_csv("https://raw.githubusercontent.com/marceloreis/HTI/master/PRSA_Data_20130301-20170228/PRSA_Data_Guanyuan_20130301-20170228.csv")

dfs = {
    "aotizhongxin_df": aotizhongxin_df,
    "changping_df": changping_df,
    "dingling_df": dingling_df,
    "dongsi_df": dongsi_df,
    "guanyuan_df": guanyuan_df,
    "gucheng_df": gucheng_df,
    "huairou_df": huairou_df,
    "nongzhanguan_df": nongzhanguan_df,
    "shunyi_df": shunyi_df,
    "tiantan_df": tiantan_df,
    "wanliu_df": wanliu_df,
    "wanshouxigong_df": wanshouxigong_df,
}

area_list = [
    "aotizhongxin_df",
    "changping_df",
    "dingling_df",
    "dongsi_df",
    "guanyuan_df",
    "gucheng_df",
    "huairou_df",
    "nongzhanguan_df",
    "shunyi_df",
    "tiantan_df",
    "wanliu_df",
    "wanshouxigong_df",
]

def checkNormality(data):
    for column_name, column_values in data.items():
        if pd.api.types.is_numeric_dtype(column_values):
            sdev = column_values.std()
            print(f"Standard Deviation of {column_name}: {sdev}")
            p_values = stats.shapiro(column_values)
            print(f"With p-values of {column_name}: {p_values}")



def fillMissingValues(data):
    filled_data = data.copy()
    for column_name, column_values in filled_data.items():
        if pd.api.types.is_numeric_dtype(column_values):
            column_values.fillna(value=column_values.median(), inplace=True)
            print(
                f"Missing values in {column_name} after: {column_values.isna().sum()}"
            )

    return filled_data

def checkBro(chooseData):
    chooseData.info()
    print(str(chooseData).upper())
    print("Missing values\t:\t", chooseData.isna().sum())
    # print(chooseData.hist())
    checkNormality(chooseData)
    print("\n\n")

for area, df in dfs.items():
    checkBro(df)


dfs_after = {}

for area, df in dfs.items():
    df_after = fillMissingValues(df)
    dfs_after[area + '_after'] = df_after


for area, df_after in dfs_after.items():
    print(f"{area} Duplicates: {df_after.duplicated().sum()}")

print("THERE ARE NO DUPLICATES YEESSS")


def imputationOutliers(chooseData):
    for column_name, column_values in chooseData.items():
        if pd.api.types.is_numeric_dtype(column_values):
            q1 = np.percentile(column_values, 25)
            q3 = np.percentile(column_values, 75)
            iqr = q3 - q1
            upperBound = q3 + 3 * iqr
            lowerBound = q1 - 3 * iqr

            if iqr > 0:
                outliers = column_values[
                    (column_values < lowerBound) | (column_values > upperBound)
                ]
                print(
                    column_name,
                    "IQR   \t:",
                    iqr,
                    "                  \t=====>>>>>",
                    column_name,
                    "outliers : ",
                    outliers.count(),
                )

                chooseData[column_name] = np.where(
                    column_values < lowerBound, lowerBound, column_values
                )
                chooseData[column_name] = np.where(
                    column_values > upperBound, upperBound, column_values
                )

            else:
                pass
    print("\n")

dfs_outlier_imputed = {}

for area, df in dfs_after.items():
    print(area)
    imputationOutliers(df)
    dfs_outlier_imputed[area] = df


dfs_cleaned = {}
for area, df in dfs_outlier_imputed.items():
    modeWindDirection = df["wd"].mode()[0]
    df["wd"].fillna(modeWindDirection, inplace=True)
    print(area, " Missing Values \t: ", df["wd"].isna().sum())
    dfs_cleaned[area] = df


dfs_cleaned['aotizhongxin_df_after'].info()

dfs_cleaned['aotizhongxin_df_after'].describe()


all_data = pd.concat(dfs_cleaned.values())



def calculateAqi(pollutant, averageDaily):
    global aqiTable
    aqiTable = {
        'PM2.5': {
            50: (0, 12),
            100: (12.1, 35.4),
            150: (35.5, 55.4),
            200: (55.5, 150.4),
            300: (150.5, 250.4),
            400: (250.5, 350.4),
            500: (350.5, 500.4)
        },
        'PM10': {
            50: (0, 50),
            100: (51, 100),
            250: (101, 250),
            350: (251, 350),
            420: (351, 420),
            500: (421, 500)
        },
        'SO2': {
            50: (0, 35),
            100: (36, 75),
            150: (76, 185),
            200: (186, 304),
            300: (305, 604),
            400: (605, 804),
            500: (805, 1004)
        },
        'CO': {
            50: (0, 4.4),
            100: (4.5, 9.4),
            150: (9.5, 12.4),
            200: (12.5, 15.4),
            300: (15.5, 30.4),
            400: (30.5, 40.4),
            500: (40.5, 50.4)
        },
        'O3': {
            50: (0, 54),
            100: (55, 70),
            150: (71, 85),
            200: (86, 105),
            300: (106, 200),
            400: (201, 504),
            500: (505, 604)
        },
        'NO2': {
            50: (0, 53),
            100: (54, 100),
            150: (101, 360),
            200: (361, 649),
            300: (650, 1249),
            400: (1250, 1649),
            500: (1650, 2049)
        },
    }

    if pollutant in aqiTable:
        for aqiValue, (low, high) in aqiTable[pollutant].items():
            if low <= averageDaily <= high:
                result = ((averageDaily - low) / (high - low)) * (aqiValue - 0) + 0
                return result

        maxAqi = max(aqiTable[pollutant].keys())
        if averageDaily > high:
            scaled_result = maxAqi + ((averageDaily - high) / (averageDaily - high)) * (500 - maxAqi)
            return scaled_result
        else:
            return 500
    else:
        print("Input pollutant between PM2.5, PM10, CO, SO, NO2, O3")
        return None


def aqiCategory(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 199:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 299:
        return "Unhealthy"
    elif aqi <= 399:
        return "Very Unhealthy"
    else:
        return "Hazardous"


group_byyear = all_data.groupby(by="year").agg({
    "PM2.5": "mean",
    "PM10": "mean",
    "SO2":"mean",
    "NO2":"mean",
    "CO":"mean",
    "O3":"mean",
})


group_byaqi = all_data.groupby(by="year").agg({
    "PM2.5": lambda x: calculateAqi("PM2.5", x.mean()),
    "PM10": lambda x: calculateAqi("PM10",x.mean()),
    "SO2": lambda x: calculateAqi("SO2",x.mean()),
    "NO2": lambda x: calculateAqi("NO2",x.mean()),
    "CO": lambda x: calculateAqi("CO",x.mean()),
    "O3": lambda x: calculateAqi("O3",x.mean())
})


def calculatePollutantDaily(chooseData,pollutant,day,month,year):
    global sumDaily
    global averageDaily

    #CHECK DATA TYPE
    if type(pollutant) != str:
        print("Input string in pollutant")

    if type(day) != int:
        print("Input int in day")

    if type(month) != int:
        print("Input int in day month")

    if type(year) != int:
        print("Input int in day year")
    elif len(str(year)) != 4:
        print("Input 4 digit in year")


    sumDaily = chooseData[pollutant][(chooseData["year"] == year) & (chooseData["month"] == month) & (chooseData["day"] == day)].sum()
    averageDaily = chooseData[pollutant][(chooseData["year"] == year) & (chooseData["month"] == month) & (chooseData["day"] == day)].mean()
    aqi = calculateAqi(pollutant, averageDaily)

    return sumDaily, averageDaily, aqi, aqiCategory(aqi)


def calculatePollutantMonthly(chooseData, pollutant, month, year):
    global averageMonthly
    global sumMonthly
    # CHECK DATA TYPE
    if type(pollutant) != str:
        print("Input string in pollutant")

    if type(month) != int:
        print("Input int in day month")

    if type(year) != int:
        print("Input int in day year")
    elif len(str(year)) != 4:
        print("Input 4 digit in year")

    sumMonthly = chooseData[pollutant][(chooseData["year"] == year) & (chooseData["month"] == month)].sum()
    averageMonthly = chooseData[pollutant][(chooseData["year"] == year) & (chooseData["month"] == month)].mean()
    aqi = calculateAqi(pollutant, averageMonthly)

    return sumMonthly, averageMonthly,aqi, aqiCategory(aqi)


calculatePollutantMonthly(aotizhongxin_df, "PM2.5", 2, 2014)


def calculatePollutantYearly(chooseData, pollutant, year):
    global sumYearly
    global averageYearly
    if type(pollutant) != str:
        print("Input string in pollutant")

    if type(year) != int:
        print("Input int in day year")
    elif len(str(year)) != 4:
        print("Input 4 digit in year")

    sumYearly = chooseData[pollutant][(chooseData["year"] == year)].sum()
    averageYearly = chooseData[pollutant][(chooseData["year"] == year)].mean()
    aqi = calculateAqi(pollutant, averageYearly)

    return sumYearly, averageYearly,aqi, aqiCategory(aqi)

st.markdown("<h1 style='text-align: center;'>Air Quality District of Beijing</h1>", unsafe_allow_html=True)
st.write("<h4 style='text-align: center;'>Preliminary Risk Screen Assessments (PRSAs)</h4>", unsafe_allow_html=True)
st.write("<h4 style='text-align: center;'>2013 - 2017</h4><br /><br /><br /><hr />", unsafe_allow_html=True)

plt.style.use('dark_background')
with st.sidebar:
    st.subheader("AQI in daily")
    st.text("Choose the Data")


    SelectCity = st.selectbox("Select City",
        [
            "aotizhongxin_df_after",
            "changping_df_after",
            "dingling_df_after",
            "dongsi_df_after",
            "guanyuan_df_after",
            "gucheng_df_after",
            "huairou_df_after",
            "nongzhanguan_df_after",
            "shunyi_df_after",
            "tiantan_df_after",
            "wanliu_df_after",
            "wanshouxigong_df_after",
        ]
    )



    data = st.selectbox(
        "Select Data",
        [
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "wd",
            "WSPM",
        ])
    
    selectDate = st.date_input(
        label="Date",
        min_value=datetime.date(2013, 3, 1),
        max_value=datetime.date(2017, 2, 28),
        value=datetime.date(2013, 3, 1),
    )


    st.write("Data ", data, "in ", selectDate, " at ", SelectCity," : ")

    selected_df = dfs_cleaned[SelectCity]

    day = selectDate.day
    month = selectDate.month
    year = selectDate.year


    if selected_df is not None:
        st.write(selected_df[data][(selected_df["year"] == year) & (selected_df["month"] == month) & (selected_df["day"] == day)].describe(include="all"))
        _, _, _, aqi_category = calculatePollutantDaily(selected_df, data, day, month, year)
        st.write("AQI Category :", aqi_category)
    else:
        st.write("Invalid selection. Please choose a valid city.")


col1, col2, col3, = st.columns(3)

st.subheader("Overall Average per Pollutant")
with col1:
    st.write("Average Pollutan PM2.5 in Beijing")
    st.code(group_byyear["PM2.5"].mean())
    st.write("Average Pollutan NO2 in Beijing")
    st.code(group_byyear["NO2"].mean())
with col2:
    st.write("Average Pollutan PM10 in Beijing")
    st.code(group_byyear["PM10"].mean())
    st.write("Average Pollutan CO in Beijing")
    st.code(group_byyear["CO"].mean())
with col3:
    st.write("Average Pollutan SO2 in Beijing")
    st.code(group_byyear["SO2"].mean())
    st.write("Average Pollutan SO2 in Beijing")
    st.code(group_byyear["SO2"].mean())

fig, ax = plt.subplots(figsize=(10,8))
x_axis = group_byyear.index
y_axis = group_byyear.values
keysyear = group_byyear.keys()
ax.set_title("Average Concentration by Year", fontsize=20)
ax.plot(x_axis,y_axis)
ax.legend(keysyear, loc ='right')
ax.set_xlabel("Year")
ax.set_ylabel("Concentration of pollutant")
st.pyplot(fig)


col1, col2, col3, = st.columns(3)

st.subheader("Overall Average per Pollutant")
with col1:
    st.write("AQI PM2.5 in Beijing")
    st.code(group_byaqi["PM2.5"].mean())
    st.write("AQI NO2 in Beijing")
    st.code(group_byaqi["NO2"].mean())
with col2:
    st.write("AQI PM10 in Beijing")
    st.code(group_byaqi["PM10"].mean())
    st.write("AQI CO in Beijing")
    st.code(group_byaqi["CO"].mean())
with col3:
    st.write("AQI SO2 in Beijing")
    st.code(group_byaqi["SO2"].mean())
    st.write("Average SO2 in Beijing")
    st.code(group_byaqi["SO2"].mean())


fig, ax = plt.subplots(figsize=(10,8))
x_axis = group_byaqi.index
y_axis = group_byaqi.values
keysaqi = group_byaqi.keys()
ax.set_title("Average AQI by Year", fontsize=20)
ax.plot(x_axis,y_axis)
ax.set_xlabel("Year")
ax.legend(keysaqi, loc ='right')
ax.set_ylabel("AQI of pollutant")
st.pyplot(fig)


st.subheader("AQI categories in each cities by pollutant")
pollutant = st.selectbox(
    "Select Pollutant",[
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",

    ])

cityLabel = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']





cleaned_data_key = [key for key in dfs_cleaned.keys() if 'after' in key][0]

avg = {}
for area, df in dfs_cleaned.items():
    average = df[pollutant].mean()
    aqi = calculateAqi(pollutant,average)
    if aqi is not None:
        avg[area] = {
            'Average': average,
            'Aqi':aqi,
            'Category':aqiCategory(aqi)
        }
    else:
        st.write("Error calculation AQI for", pollutant,' in ',area)

avg_df = pd.DataFrame.from_dict(avg, orient='index')

cmap = plt.get_cmap('PuRd')

colors = {
    'Good': cmap(0.1),
    'Moderate': cmap(0.39),
    'Unhealthy for Sensitive Groups': cmap(0.5),
    'Unhealthy': cmap(0.7),
    'Very Unhealthy': cmap(0.9),
    'Hazardous': cmap(1.0)
}

bar_colors = [colors[category] for category in avg_df['Category']]

fig, ax = plt.subplots()
ax.bar(x=cityLabel, height=avg_df['Aqi'], color=bar_colors)
ax.set_title("AQI {} Comparison".format(pollutant), loc="center", fontsize=20)
ax.set_xlabel(None)
ax.set_ylabel("Average Pollutant Concentrate")
plt.xticks(rotation=45)
plt.xticks(range(len(cityLabel)), cityLabel, ha='right')
legend_handles = [plt.Rectangle((0,0),1,1, color=colors[category]) for category in colors.keys()]
legend_labels = colors.keys()
ax.legend(legend_handles,legend_labels,loc='lower center', bbox_to_anchor=(0.5,-0.46),ncol=len(colors))
st.pyplot(fig)


st.markdown("<br/><h3 style='text-align: center;'>Average Pollutant Concentration Comparison for Every City</h4>", unsafe_allow_html=True)





col3, col4,col5 = st.columns(3)
dict_col1 = {
    'aotizhongxin' : dfs_cleaned['aotizhongxin_df_after'],
    'dongsi': dfs_cleaned['dongsi_df_after'],
    'huairou': dfs_cleaned['huairou_df_after'],
    'tiantan': dfs_cleaned['tiantan_df_after']

    }

dict_col2 = {
    'changping': dfs_cleaned['changping_df_after'],
    'guanyuan': dfs_cleaned['guanyuan_df_after'],
    'nongzhanguan': dfs_cleaned['nongzhanguan_df_after'],
    'wanliu': dfs_cleaned['wanliu_df_after']

}

dict_col3 = {
    'dingling': dfs_cleaned['dingling_df_after'],
    'gucheng': dfs_cleaned['gucheng_df_after'],
    'shunyi': dfs_cleaned['shunyi_df_after'],
    'wanshouxigong': dfs_cleaned['wanshouxigong_df_after']
}

with col3:
    for area, df in dict_col1.items():
        df_cleaned = df.agg({
            "PM2.5": "mean",
            "PM10": "mean",
            "SO2":"mean",
            "NO2":"mean",
            "CO":"mean",
            "O3":"mean",
        })

        fig, ax = plt.subplots(figsize=(6,6))
        pollutants = df_cleaned.index
        mean_concentrations = df_cleaned.values
        ax.bar(pollutants, mean_concentrations, color=cmap(0.2))
        ax.set_title("In {} ".format(df['station'][0]))
        ax.set_xlabel("Pollutant")
        ax.set_ylabel("Average Concentration")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)


with col4:
    for area, df in dict_col2.items():
        df_cleaned = df.agg({
            "PM2.5": "mean",
            "PM10": "mean",
            "SO2":"mean",
            "NO2":"mean",
            "CO":"mean",
            "O3":"mean",
        })

        fig, ax = plt.subplots(figsize=(6,6))
        pollutants = df_cleaned.index
        mean_concentrations = df_cleaned.values
        ax.bar(pollutants, mean_concentrations, color=cmap(0.2))
        ax.set_title("In {} ".format(df['station'][0]))
        ax.set_xlabel("Pollutant")
        ax.set_ylabel("Average Concentration")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)



with col5:
    for area, df in dict_col3.items():
        df_cleaned = df.agg({
            "PM2.5": "mean",
            "PM10": "mean",
            "SO2":"mean",
            "NO2":"mean",
            "CO":"mean",
            "O3":"mean",
        })

        fig, ax = plt.subplots(figsize=(6,6))
        pollutants = df_cleaned.index
        mean_concentrations = df_cleaned.values
        ax.bar(pollutants, mean_concentrations, color=cmap(0.2))
        ax.set_title("In {} ".format(df['station'][0]))
        ax.set_xlabel("Pollutant")
        ax.set_ylabel("Average Concentration")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)



st.caption('Copyright (c) 2023')