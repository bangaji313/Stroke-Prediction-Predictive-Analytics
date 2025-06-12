# ===================================================================================
# Proyek Predictive Analytics: Prediksi Risiko Stroke
# Nama: Maulana Seno Aji Yudhantara
# Email: senoaji115@gmail.com
# ID Dicoding: bang_aji
# Cohort ID Coding Camp: MC117D5Y1789
# File: stroke_prediction_script.py
#
# Deskripsi:
# Skrip ini mengimplementasikan alur kerja machine learning lengkap untuk
# memprediksi risiko stroke. Prosesnya meliputi pemuatan data, persiapan data
# (imputasi, encoding, scaling), penanganan data tidak seimbang (SMOTE),
# pelatihan model Logistic Regression, dan evaluasi performa model.
# ===================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path):
    """Memuat dataset dari file CSV."""
    print(f"Memuat data dari {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Melakukan pra-pemrosesan data awal."""
    print("Melakukan pra-pemrosesan data...")
    # Menghapus kolom ID yang tidak relevan
    df = df.drop(columns=['id'])
    # Menghapus baris dengan gender 'Other' yang merupakan anomali
    df = df[df['gender'] != 'Other']
    return df

def main():
    """Fungsi utama untuk menjalankan seluruh alur kerja machine learning."""
    
    # 1. Memuat Data
    file_path = 'healthcare-dataset-stroke-data.csv'
    df = load_data(file_path)
    
    # 2. Pra-pemrosesan Awal
    df_processed = preprocess_data(df)
    
    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df_processed.drop(columns=['stroke'])
    y = df_processed['stroke']
    
    # 4. Membagi Data menjadi Data Latih dan Uji
    print("Membagi data menjadi set latih dan uji...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Membuat Pipeline untuk Pra-pemrosesan Fitur
    # Pipeline ini akan menangani imputasi, scaling, dan encoding
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Transformer untuk fitur numerik: imputasi median lalu scaling
    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Transformer untuk fitur kategorikal: imputasi modus lalu one-hot encoding
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Menggabungkan transformer dengan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 6. Membuat Pipeline Machine Learning Lengkap
    # Pipeline ini menggabungkan pra-pemrosesan, SMOTE, dan model
    print("Membangun pipeline machine learning...")
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # 7. Melatih Model
    print("Melatih model Logistic Regression...")
    model_pipeline.fit(X_train, y_train)
    
    # 8. Mengevaluasi Model
    print("Mengevaluasi model pada data uji...")
    y_pred = model_pipeline.predict(X_test)
    
    # 9. Menampilkan Hasil Evaluasi
    print("\n" + "="*40)
    print("      HASIL EVALUASI MODEL")
    print("="*40)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (untuk kelas 1): {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (untuk kelas 1): {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score (untuk kelas 1): {f1_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*40)
    print("Skrip selesai dijalankan.")

if __name__ == '__main__':
    main()