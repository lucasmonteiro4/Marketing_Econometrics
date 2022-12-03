import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

pd.set_option('display.max_columns', 200)


train = pd.read_csv('Data/Train_transactions.csv')
test = pd.read_csv('Data/Test_transactions.csv')
y = pd.read_csv('Data/Train_customers_repurchase.csv')

train.head()

train['card_sub'] = np.where(train['card_subscription'].isnull(), 0, 1)
train.transaction_date = pd.to_datetime(train.transaction_date)
train.dtypes

s = train.transaction_date.max() - train.transaction_date
train['last_visit'] = s.dt.days

train_ok = (
    train
    .groupby('id_client')
    .agg(
        last_visit = ('last_visit', 'min'),
        nb_purchases = ('id_client', 'count'),
        item_count = ('item_count', 'mean'),
        gross_amount = ('gross_amount', 'mean'),
        discount_amount = ('discount_amount', 'mean'),
        basket_value = ('basket_value', 'mean'),
        payment_gift = ('payment_gift', 'mean'),
        payment_cheque = ('payment_cheque', 'mean'),
        payment_cash = ('payment_cash', 'mean'),
        payment_card = ('payment_card', 'mean'),
        card_sub = ('card_sub', 'max')
    )
)

train_ok = train_ok.fillna(0)

cible = y.sort_values('id_client')['repurchase']

test['card_sub'] = np.where(test['card_subscription'].isnull(), 0, 1)

test.transaction_date = pd.to_datetime(test.transaction_date).dt.date

s = test.transaction_date.max() - test.transaction_date
test['last_visit'] = s.dt.days

test_ok = (
    test
    .groupby('id_client')
    .agg(
        last_visit = ('last_visit', 'min'),
        nb_purchases = ('id_client', 'count'),
        item_count = ('item_count', 'mean'),
        gross_amount = ('gross_amount', 'mean'),
        discount_amount = ('discount_amount', 'mean'),
        basket_value = ('basket_value', 'mean'),
        payment_gift = ('payment_gift', 'mean'),
        payment_cheque = ('payment_cheque', 'mean'),
        payment_cash = ('payment_cash', 'mean'),
        payment_card = ('payment_card', 'mean'),
        card_sub = ('card_sub', 'max')
    )
)

test_ok = test_ok.fillna(0)





train_ok_std = train_ok.copy()
train_ok_std[['last_visit', 'nb_purchases', 'item_count', 'gross_amount', 'discount_amount', 'basket_value', 'payment_gift', 'payment_cheque', 'payment_cash', 'payment_card']] = StandardScaler().fit_transform(train_ok[['last_visit', 'nb_purchases', 'item_count', 'gross_amount', 'discount_amount', 'basket_value', 'payment_gift', 'payment_cheque', 'payment_cash', 'payment_card']])

test_ok_std = test_ok.copy()
test_ok_std[['last_visit', 'nb_purchases', 'item_count', 'gross_amount', 'discount_amount', 'basket_value', 'payment_gift', 'payment_cheque', 'payment_cash', 'payment_card']] = StandardScaler().fit_transform(test_ok[['last_visit', 'nb_purchases', 'item_count', 'gross_amount', 'discount_amount', 'basket_value', 'payment_gift', 'payment_cheque', 'payment_cash', 'payment_card',]])

rf_std_class = RandomForestClassifier(max_depth=4, min_samples_split=10,n_estimators=100, bootstrap=True, oob_score=True)
rf_std_class.fit(train_ok_std, cible)

predict_rf_std = rf_std_class.predict_proba(test_ok_std)

test_ok_std_reset = test_ok_std.reset_index()
test_ok_reset = test_ok.reset_index()

submission = pd.DataFrame({
    'Id': test_ok_std_reset.id_client,
    'Expected': predict_rf_std[:, 1]
})

submission.to_csv('submission_nb.csv', index = False, sep = ',', decimal = '.')

submission = submission.sort_values('Expected')

last_ind = int(submission.shape[0]*0.10)
last_ind
last10 = submission[-last_ind:]
sns.histplot(last10["Expected"])

last10

test_ok_reset["Customers class"] = np.where(test_ok_reset.id_client.isin(last10["Id"]) , 'top customers', 'else')
test_ok_reset[test_ok_reset["Customers class"]=='top customers']

g =sns.kdeplot(data=test_ok_reset, x="item_count", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g.set(xlim=(0, 200))

g = sns.kdeplot(data=test_ok_reset, x="gross_amount", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g.set(xlim=(0, 200))

g =sns.kdeplot(data=test_ok_reset, x="basket_value", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g.set(xlim=(0, 200))

g = sns.kdeplot(data=test_ok_reset, x="discount_amount", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g.set(xlim=(0, 50))

g = sns.kdeplot(data=test_ok_reset, x="payment_cheque", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g

g = sns.kdeplot(data=test_ok_reset, x="payment_card", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g

g = sns.kdeplot(data=test_ok_reset, x="payment_cash", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g

g = sns.kdeplot(data=test_ok_reset, x="payment_gift", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g

g = sns.kdeplot(data=test_ok_reset, x="nb_purchases", hue="Customers class", cut=0, fill=False, common_norm=False, alpha=1)
g.set(xlim=(0, 250))

sns.barplot(data=test_ok_reset, x="card_sub", y="discount_amount", hue="Customers class")