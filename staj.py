
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Veri setini bilgisayardaki klasörden okuma
train_data = pd.read_csv('train.csv', header=0, names=['classid', 'title', 'desc'])
test_data = pd.read_csv('test.csv', header=0, names=['classid', 'title', 'desc'])

# Metinleri birleştirme
train_data['text'] = train_data['title'] + ' ' + train_data['desc']
test_data['text'] = test_data['title'] + ' ' + test_data['desc']
print (train_data.head())
import re
import string

def clean_text(text):
    # Küçük harfe çevirir
    text = text.lower()
    # Noktalama işaretlerini kaldırır
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text

# Temizleme işlemini train ve test setine uygulama
train_data['text'] = train_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)

print("Temizlenmiş veriden örnek:")
print(train_data['text'].head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. TF-IDF Vektörleştiriciyi tanımla ve eğit (fit_transform)
vectorizer = TfidfVectorizer(max_features=5000) # Çok büyük olmaması için en önemli 5000 kelimeyi alalım
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

y_train = train_data['classid']
y_test = test_data['classid']

print("TF-IDF matrisi başarıyla oluşturuldu!")

# 2. Model 1: Naive Bayes Modelini Eğitme
print("\n--- Naive Bayes Modeli Eğitiliyor ---")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# 3. Model 2: Logistic Regression Modelini Eğitme
print("\n--- Logistic Regression Modeli Eğitiliyor ---")
# AG News veri seti büyük olduğu için iterasyon sınırını (max_iter) artırıyoruz
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("İlk iki model başarıyla eğitildi ve tahminler yapıldı!")
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Naive Bayes Değerlendirmesi
print("\n--- Naive Bayes Sonuçları ---")
print(classification_report(y_test, nb_pred))

# 2. Logistic Regression Değerlendirmesi
print("\n--- Logistic Regression Sonuçları ---")
print(classification_report(y_test, lr_pred))

# 3. Confusion Matrix (Karmaşıklık Matrisi) Görselleştirme - Logistic Regression için
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()
