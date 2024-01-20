####################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
#####################################

# 1. Verinin Hazırlanması (Data Prepretation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'YE Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması

###################################
# 1. Verinin Hazırlanması (Data Prepretation)
###################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x : "%.4f" % x)



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable]<low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable]<up_limit), variable] = up_limit



df_ = pd.read_excel(r"C:\Users\kkubi\OneDrive\Masaüstü\CRM_Analytics\online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

########################
# Veri Ön İşleme
########################

df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011,12,11)


#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency : Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T : Müşterinin Yaşı. (Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency : Tekrar eden toplan satın alma sayısı (frequency > 1)
# monetary_value : satın alma başına ortalama kazanç


cltv_df = df.groupby("Customer ID").agg({
                                        "InvoiceDate":[lambda InvoiceDate:(InvoiceDate.max()-InvoiceDate.min()).days,
                                                        lambda date:(today_date-date.min()).days],
                                        "Invoice":lambda x : x.nunique(),
                                        "TotalPrice": lambda x: x.sum()})




cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency","T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7



##########################################
# 2.BG-NBD Modelinin Kurulması
#########################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])


############################
# 1 Hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
############################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)



cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])


###############################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
###############################

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])

bgf.predict(4,
              cltv_df["frequency"],
              cltv_df["recency"],
              cltv_df["T"]).sum()



###############################
# Tahmin sonuçlarının Değerlendirilmesi?
##############################

plot_period_transactions(bgf)
plt.show()


################################
# 3. Gamma-Gamma Modelinin Kurulması
###############################


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).head(10)


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"])



###########################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması
###########################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,
                                   freq="M",
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()


cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head()



#####################################
# 5. CLTV'ye göre Segmentlerin Oluşturulması
######################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])


cltv_final.groupby("segment").agg(
    {"count","mean","sum"})


#####################################
# 6. Çalışmanın Fonksiyonlaştırılması
######################################


def create_cltv_p(dataframe, month =3):
    # 1. Veri ön işleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"]>0]
    dataframe = dataframe[dataframe["Price"]>0]
    replace_with_thresholds(dataframe,"Quantity")
    replace_with_thresholds(dataframe,"Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011,12,11)

    cltv_df = dataframe.groupby("Customer ID").agg(
        {"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                        lambda x :(today_date-x.min()).days],
         "Invoice":lambda x : x.nunique(),
         "TotalPrice": lambda x : x.sum()})
    
    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df["frequency"]>1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])
    
    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])
    
    cltv_df["expected_purc_3_months"] = bgf.predict(12,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])
    
    #3.Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                                 cltv_df["monetary"])
    
    #4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması
    cltv = ggf.customer_lifetime_value(
        bgf,
        cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"],
        cltv_df["monetary"],
        time= month,
        freq= "M",
        discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how= "left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D","C","B","A"])

    return cltv_final


df = df_.copy()

cltv_final2 = create_cltv_p(df)














