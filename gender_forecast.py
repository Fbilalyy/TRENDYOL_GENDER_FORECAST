import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from lightgbm import early_stopping, log_evaluation as lg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from collections import Counter
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
def train_and_evaluate_lightgbm_classifier_with_cv(train_folder, test_folder, cv_folds=5):
    auc_scores = []

    train_files = sorted(os.listdir(train_folder))

    for file_name in train_files:
        if not file_name.endswith(".csv"):
            continue

        print(f"Processing: {file_name}")

        train_path = os.path.join(train_folder, file_name)
        test_path = os.path.join(test_folder, file_name)

        if not os.path.exists(test_path):
            print(f"Test file {file_name} not found. Skipping.")
            continue

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if train_df.empty or test_df.empty:
                print(f"{file_name} is empty. Skipping.")
                continue

            # Ayrıştır
            X = train_df.drop(columns=['gender'], errors='ignore')
            y = train_df['gender']

            # Label Encoding
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Cross-validation başlat
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_aucs = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMClassifier(
                    objective='binary',
                    boosting_type='gbdt',
                    n_estimators=1000,
                    random_state=42
                )

                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(10)]
                )

                y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                fold_auc = roc_auc_score(y_val_fold, y_val_pred_proba)
                fold_aucs.append(fold_auc)

                print(f"  Fold {fold} AUC: {fold_auc:.4f}")

            mean_cv_auc = sum(fold_aucs) / len(fold_aucs)
            print(f"✅ CV AUC (avg of {cv_folds} folds): {mean_cv_auc:.4f}")
            auc_scores.append(mean_cv_auc)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    if auc_scores:
        print(f"\n🔚 Average CV AUC across all files: {sum(auc_scores) / len(auc_scores):.4f}")
    else:
        print("No AUC scores were calculated.")
def train_and_evaluate_lightgbm_classifier(train_folder, test_folder):
    auc_scores = []

    train_files = sorted(os.listdir(train_folder))

    for file_name in train_files:
        if not file_name.endswith(".csv"):
            continue

        print(f"Processing: {file_name}")

        # Train ve Test dosyaların654ı oku
        train_path = os.path.join(train_folder, file_name)
        test_path = os.path.join(test_folder, file_name)

        if not os.path.exists(test_path):
            print(f"Test file {file_name} not found. Skipping.")
            continue

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if train_df.empty or test_df.empty:
                print(f"{file_name} is empty. Skipping.")
                continue

            # 'gender' kolonunu ayır
            X_train = train_df.drop(columns=['gender'], errors='ignore')
            y_train = train_df['gender']

            X_test = test_df.drop(columns=['gender'], errors='ignore')
            y_test = test_df['gender']

            # Label Encoding (sadece object kolonlara)
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_test[col].dtype == 'object':
                    le = LabelEncoder()
                    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
                    le.fit(combined)
                    X_train[col] = le.transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))

            # LightGBM Classifier
            model = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                n_estimators=1000,
                random_state=42
            )

            # Modeli eğit
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(10)]  # early_stopping_rounds burada çağrılır
            )
            if len(set(y_test)) < 2:
                print(f"Skipping {csv_name}: Only one class in y_test")
                continue

            # Tahminler (olasılık olarak)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # sadece 1 sınıf olasılığı

            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC Score: {auc:.4f}")

            # AUC Skoru


        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    if auc_scores:
        print(f"\nAverage AUC across all files: {sum(auc_scores) / len(auc_scores):.4f}")
    else:
        print("No AUC scores were calculated.")
def plot_gender_distribution_by_feature(df, features, target_col="gender"):
    """
    Her feature için distinct value'lara göre gender ortalamasını scatter plot ile gösterir.
    Bütün feature'ları tek bir büyük figure içine toplar.
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Satır sayısı otomatik ayarlanır

    plt.figure(figsize=(20, 5 * n_rows))

    for idx, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, idx + 1)

        # Ortalama gender değerini bul
        grouped = df.groupby(feature)[target_col].mean()

        # Nokta grafiği
        plt.scatter(grouped.index.astype(str), grouped.values, alpha=0.7)
        plt.xticks(rotation=90)
        plt.title(f"Gender Mean by {feature}")
        plt.ylabel("Gender Mean (1=Female, 0=Male)")
        plt.xlabel(f"{feature}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
def clean_excel_data(df):
    """
    Excel/CSV verisini tam kapsamlı temizleyen ve düzenleyen fonksiyon.
    - Boş satır/kolon temizler
    - Duplicate kayıtları kaldırır
    - Veri tiplerini düzeltir
    - Outlierları işaretler
    - İsimlendirmeyi düzeltir
    - Gereksiz kolonları kaldırır
    """

    print("İlk veri şekli:", df.shape)

    # 1. Tamamen boş kolonları sil
    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        print(f"Tamamen boş {len(empty_cols)} kolon silindi: {list(empty_cols)}")
        df = df.drop(columns=empty_cols)

    # 2. Tamamen boş satırları sil
    before_rows = df.shape[0]
    df = df.dropna(how='all')
    after_rows = df.shape[0]
    print(f"Tamamen boş {before_rows - after_rows} satır silindi.")

    # 3. Duplicate satırları sil
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    after_rows = df.shape[0]
    print(f"{before_rows - after_rows} duplicate satır silindi.")

    # 4. Boş değerleri doldur
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna('Unknown')

    # 5. İsimlendirmeyi standardize et
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]

    # 6. Veri tiplerini düzelt
    for col in df.columns:
        if 'date' in col or 'timestamp' in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        elif df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                df[col] = df[col].astype('category')

    # 7. Sadece tek değer alan kolonları sil
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"Sadece tek değer içeren {len(constant_cols)} kolon silindi: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # 8. Sayısal kolonlardaki uç değerleri (outlier) tespit ve işaretleme
    outlier_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"{col} kolonunda {outliers} uç değer bulundu.")
            outlier_columns.append(col)

    if outlier_columns:
        print(f"Uç değer bulunan kolonlar: {outlier_columns}")

    # 9. Mantıksal hataları düzelt
    if 'sellingprice' in df.columns:
        negative_prices = (df['sellingprice'] < 0).sum()
        if negative_prices > 0:
            print(f"{negative_prices} adet negatif fiyat sıfırlandı.")
            df.loc[df['sellingprice'] < 0, 'sellingprice'] = 0

    if 'gender' in df.columns:
        wrong_genders = (~df['gender'].isin([0, 1])).sum()
        if wrong_genders > 0:
            print(f"{wrong_genders} adet hatalı gender değeri Unknown yapıldı.")
            df.loc[~df['gender'].isin([0, 1]), 'gender'] = 'Unknown'

    print("Son veri şekli:", df.shape)
    print("✅ Data cleaning tamamlandı.")

    return df
def clean_all_csv_in_folder(input_folder, output_folder):
    """
    Belirtilen klasördeki tüm CSV dosyalarını temizler ve yeni bir klasöre kaydeder.
    Her dosyanın sonunda kaç satır kaldığı yazdırılır.
    """
    # Eğer çıktı klasörü yoksa oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Klasördeki tüm dosyalar
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    print(f"{len(files)} CSV dosyası bulundu. İşleme başlanıyor...\n")

    for file in files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        print(f"📂 Şu dosya işleniyor: {file} ({df.shape[0]} satır)")

        # Temizle
        cleaned_df = clean_excel_data(df)

        # Yeni yolu oluştur
        output_path = os.path.join(output_folder, file)
        cleaned_df.to_csv(output_path, index=False)

        print(f"✅ Kaydedildi: {output_path} ({cleaned_df.shape[0]} satır)\n")

    print("🎯 Tüm dosyalar başarıyla temizlendi ve kaydedildi!")

def count_total_rows(folder_path):
    total_rows = 0
    file_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"{file_name} → ❌ Dosya boş (0 satır)")
                    continue  # Boş dosyayı atla
                num_rows = len(df)
                print(f"{file_name} → {num_rows} satır")
                total_rows += num_rows
                file_count += 1
            except pd.errors.EmptyDataError:
                print(f"{file_name} → ❌ HATA: Dosya tamamen boş! Atlandı.")
                continue

    print("\nToplam dosya sayısı (boş olmayanlar):", file_count)
    print("Tüm dosyalar için toplam satır sayısı:", total_rows)
    return total_rows


def train_model(df, features, target='gender'):
    X = df[features]
    y = df[target]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForest seçtik çünkü:
    # - Feature sayısı 18
    # - Veri yapısı karışık (numerik + kategorik karışımı)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc


# -- Ana Loop: Tüm CSV'ler İçin --
def process_all_csvs(folder_path):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"✅ İşleniyor: {filename}")
            try:
                df = pd.read_csv(file_path)
                df = smart_clean(df)

                if df.empty:
                    print(f"🚨 HATA: {filename} tamamen boş kaldı temizleme sonrası.")
                    continue

                # Feature listesi (örnek: tüm kolonlar - gender)
                feature_columns = [col for col in df.columns if col != 'gender']

                acc = train_model(df, feature_columns)
                results[filename] = acc
                print(f"🎯 {filename} başarıyla işlendi. Accuracy: {acc:.2f}")

            except Exception as e:
                print(f"🚨 HATA: {filename} dosyası işlenemedi. Sebep: {e}")

    return results
# Tüm önemli kolonları listele
def count_nan_inf_per_column_in_folder(folder_path):
    report = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)

                print(f"\n📄 Dosya: {filename}")

                for col in df.columns:
                    nan_count = df[col].isna().sum()
                    inf_count = np.isinf(df[col]).sum() if np.issubdtype(df[col].dtype, np.number) else 0

                    print(f"    ➔ Kolon: {col} | NaN Sayısı: {nan_count} | Inf Sayısı: {inf_count}")

            except Exception as e:
                print(f"🚨 Hata: {filename} dosyası işlenemedi. Sebep: {e}")

    return pd.DataFrame(report)
def train_and_predict_gender(train_folder, test_folder, output_folder):
    important_features = [
        'contentid', 'user_action', 'sellingprice', 'product_name', 'brand_id',
        'brand_name', 'businessunit', 'product_gender', 'category_id',
        'Level1_Category_Id', 'Level1_Category_Name', 'Level2_Category_Id', 'Level2_Category_Name',
        'Level3_Category_Id', 'Level3_Category_Name', 'unique_id', 'type'
    ]

    os.makedirs(output_folder, exist_ok=True)

    train_files = [file for file in os.listdir(train_folder) if file.endswith('.csv')]

    for file_name in train_files:
        print(f"Processing: {file_name}")

        train_path = os.path.join(train_folder, file_name)
        test_path = os.path.join(test_folder, file_name)

        # Check if corresponding test file exists
        if not os.path.exists(test_path):
            print(f"⚠️ Test file {file_name} not found, skipping.")
            continue

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        except pd.errors.EmptyDataError:
            print(f"⚠️ {file_name} is empty, skipping.")
            continue

        # Skip if train or test file is empty
        if train_df.empty or test_df.empty:
            print(f"⚠️ {file_name} is empty after reading, skipping.")
            continue

        # Identify available features
        available_features = [col for col in important_features if col in train_df.columns and col in test_df.columns]

        if len(available_features) == 0:
            print(f"⚠️ No matching important features found in {file_name}, skipping.")
            continue

        # Drop missing values
        train_df = train_df.dropna(subset=available_features + ['gender'])
        test_df = test_df.dropna(subset=available_features)

        if train_df.empty or test_df.empty:
            print(f"⚠️ {file_name} has empty train/test after dropna, skipping.")
            continue

        # Label Encoding for object columns
        combined_df = pd.concat([train_df[available_features], test_df[available_features]])
        label_encoders = {}
        for col in available_features:
            if combined_df[col].dtype == "object":
                le = LabelEncoder()
                combined_df[col] = le.fit_transform(combined_df[col].astype(str))
                label_encoders[col] = le

        # Split back
        train_encoded = combined_df.iloc[:len(train_df)]
        test_encoded = combined_df.iloc[len(train_df):]

        X_train = train_encoded
        y_train = train_df['gender']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(test_encoded)

        test_df['predicted_gender'] = predictions

        output_path = os.path.join(output_folder, file_name)
        test_df.to_csv(output_path, index=False)

    print("✅ Done training and predicting for all CSVs.")
def count_gender_one_rows(folder_path):
    total_count = 0
    total_count_0 = 0
    files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"⚠️ {file_name} is empty, skipping.")
            continue

        if 'predicted_gender' not in df.columns:
            print(f"⚠️ {file_name} does not contain 'gender' column, skipping.")
            continue

        count = (df['predicted_gender'] == 1).sum()
        total_count += count
        count_0 = (df['predicted_gender'] == 0).sum()
        total_count_0 += count_0

    print(f"✅ Total rows where gender == 1: {total_count}"
          f"✅ Total rows where gender == 0:{total_count_0}")
    return total_count, total_count_0
def select_important_features(X, y, top_n=15):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=X.columns)
    important_features = feature_importance.sort_values(ascending=False).head(top_n).index.tolist()
    return important_features
def one_hot_encode(df, categorical_cols):
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded
def fill_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError("Strategy must be 'mean' or 'median'")
def remove_outliers(df, numerical_cols, threshold=3):
    z_scores = np.abs(stats.zscore(df[numerical_cols]))
    mask = (z_scores < threshold).all(axis=1)
    cleaned_df = df[mask]
    return cleaned_df
def calculate_roc_auc(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları
    auc = roc_auc_score(y_test, y_probs)
    return auc


def train_and_predict_gender_v2(train_folder, test_folder, output_folder):
    important_features = [
        'contentid', 'user_action', 'sellingprice', 'product_name', 'brand_id',
        'brand_name', 'businessunit', 'product_gender', 'category_id',
        'Level1_Category_Id', 'Level1_Category_Name', 'Level2_Category_Id', 'Level2_Category_Name',
        'Level3_Category_Id', 'Level3_Category_Name', 'unique_id', 'type'
    ]

    os.makedirs(output_folder, exist_ok=True)

    train_files = [file for file in os.listdir(train_folder) if file.endswith('.csv')]
    auc_scores = []

    for file_name in train_files:
        print(f"Processing: {file_name}")
        if file_name == "businessunit_Kitap_&_Kırtasiye_&_Yaşam.csv":
            print(f"⚠️ Skipping {file_name} as requested.")
            continue

        train_path = os.path.join(train_folder, file_name)
        test_path = os.path.join(test_folder, file_name)

        if not os.path.exists(test_path):
            print(f"⚠️ Test file {file_name} not found, skipping.")
            continue

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        except pd.errors.EmptyDataError:
            print(f"⚠️ {file_name} is empty, skipping.")
            continue

        if train_df.empty or test_df.empty:
            print(f"⚠️ {file_name} is empty after reading, skipping.")
            continue

        available_features = [col for col in important_features if col in train_df.columns and col in test_df.columns]

        if len(available_features) == 0:
            print(f"⚠️ No matching important features found in {file_name}, skipping.")
            continue

        train_df = train_df.dropna(subset=available_features + ['gender'])
        test_df = test_df.dropna(subset=available_features)

        if train_df.empty or test_df.empty:
            print(f"⚠️ {file_name} has empty train/test after dropna, skipping.")
            continue

        combined_df = pd.concat([train_df[available_features], test_df[available_features]], axis=0)

        label_encoders = {}
        for col in available_features:
            if combined_df[col].dtype == 'object' or combined_df[col].dtype.name == 'category':
                if combined_df[col].nunique() <= 20:  # düşük unique varsa one-hot
                    dummies = pd.get_dummies(combined_df[col], prefix=col)
                    combined_df = pd.concat([combined_df.drop(columns=[col]), dummies], axis=1)
                else:
                    le = LabelEncoder()
                    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
                    label_encoders[col] = le

        train_encoded = combined_df.iloc[:len(train_df)]
        test_encoded = combined_df.iloc[len(train_df):]

        X_train = train_encoded
        y_train = train_df['gender']

        # ---- Feature Importance Based Feature Selection ----
        feature_selection_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        feature_selection_model.fit(X_train, y_train)
        importances = feature_selection_model.feature_importances_
        important_idx = np.argsort(importances)[-15:]  # En önemli 15 feature
        selected_features = X_train.columns[important_idx]

        # Final model
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train[selected_features], y_train)


        # --- ROC AUC hesaplaması için train verisini 80-20 böleceğiz ---
        X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
            X_train[selected_features], y_train, test_size=0.2, random_state=42
        )

        model_split = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model_split.fit(X_train_split, y_train_split)


        y_valid_pred_proba = model_split.predict_proba(X_valid_split)[:, 1]
        if len(np.unique(y_valid_split)) < 2:
            print(f"⚠️ Only one class present in validation split for {file_name}. Skipping AUC calculation.")
            auc_score = None
        else:
            y_valid_pred_proba = model_split.predict_proba(X_valid_split)[:, 1]
            auc_score = roc_auc_score(y_valid_split, y_valid_pred_proba)
            auc_scores.append(auc_score)

        # Eğer valid set'te sadece 1 sınıf yoksa AUC hesapla
        """if len(np.unique(y_valid_split)) > 1:
            auc = roc_auc_score(y_valid_split, y_valid_pred_proba)
            auc_scores.append(auc)
            print(f"✅ ROC AUC Score for {file_name}: {auc:.4f}")
        else:
            print(f"⚠️ Only one class in validation set for {file_name}, skipping AUC calculation.") """

        # --- Asıl test için prediction ---
        if not test_encoded.empty:
            predictions_proba = model.predict_proba(test_encoded[selected_features])[:, 1]
            predictions = (predictions_proba >= 0.5).astype(int)
            predicted_gender = ['F' if p == 1 else 'M' for p in predictions]

            test_df['predicted_gender'] = predicted_gender
            unique_ids = test_df["unique_id"].reset_index(drop=True)
            output_path = os.path.join(output_folder, file_name)
            submission_df = pd.DataFrame({
                "unique_id": unique_ids,
                "probability_female": predictions_proba,
                "gender": predicted_gender
            })

            # CSV olarak kaydet
            submission_df.to_csv(output_path, index=False)
        else:
            print(f"⚠️ Test file {file_name} after encoding is empty, skipping prediction.")

        # --- Tüm dosyalar bitince ortalama AUC sonucu ---
    if auc_scores:
        mean_auc = np.mean(auc_scores)
        print(f"🎯 Mean ROC AUC Score across all CSVs: {mean_auc:.4f}")
    else:
        print("⚠️ No AUC scores were calculated.")

    print("✅ Done training, predicting and ROC evaluation for all CSVs (v2 with AUC).")
train_folder = "/Users/fatihbilalyilmaz/Desktop/businessunit_splits_clean"
test_folder = "/Users/fatihbilalyilmaz/Desktop/businessunit_splits_test_clean"

#train_and_evaluate_lightgbm_classifier(train_folder, test_folder)
folder_path = "/Users/fatihbilalyilmaz/Desktop/businessunit_splits_predicted"
count_gender_one_rows(folder_path)
train_and_predict_gender_v2(train_folder, test_folder , "/Users/fatihbilalyilmaz/Desktop/businessunit_splits_predicted" )
folder_path = "/Users/fatihbilalyilmaz/Desktop/businessunit_splits_predicted"
all_dfs = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, file))
        all_dfs.append(df)

merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.dropna(axis=1, how='all', inplace=True)

# Unique ID bazında gruplama ve ortalama alma
grouped = merged_df.groupby("unique_id", as_index=False)["probability_female"].mean()

# 0.5 eşik ile sınıf tahmini
grouped["gender"] = grouped["probability_female"].apply(lambda x: "F" if x > 0.5 else "M")

# 🔥 Sadece gerekli 3 kolonu tut
grouped = grouped[["unique_id", "probability_female", "gender"]]

# Kaydet
grouped.to_csv("/Users/fatihbilalyilmaz/Desktop/test_prediction.csv", index=False)
file_path = "/Users/fatihbilalyilmaz/Desktop/test_prediction.csv"
df = pd.read_csv(file_path)

# Yeni satırı tanımla
new_row = {
    "unique_id": "BzGrxWeUz3",
    "probability_female": 0.54384,
    "gender": "F"
}

# Yeni satırı ekle
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Güncellenmiş dosyayı tekrar kaydet
df.to_csv(file_path, index=False)






#train_path = "/Users/fatihbilalyilmaz/Desktop/IE425_Spring25_train_data.csv"
#test_path = "/Users/fatihbilalyilmaz/Desktop/IE425_Spring25_test_data.csv"

#df = pd.read_csv(test_path)
#df["gender"] = df["gender"].map({'F': 1, 'M': 0})


#print(distinct_counts)

#input_folder = '/Users/fatihbilalyilmaz/Desktop/businessunit_splits_clean'


#clean_all_csv_in_folder(input_folder, output_folder)

"""output_folder = "/Users/fatihbilalyilmaz/Desktop/businessunit_splits_test"
os.makedirs(output_folder, exist_ok=True)

# 3. Her bir Businessunit için ayrı CSV dosyası kaydet
unique_businessunits = df['businessunit'].unique()

print(f"Toplam {len(unique_businessunits)} farklı businessunit bulundu.")

for bu in unique_businessunits:
    subset = df[df['businessunit'] == bu]

    # Dosya adlarında uygunsuz karakterleri temizle
    safe_bu_name = str(bu).replace('/', '_').replace('\\', '_').replace(' ', '_')

    output_path = os.path.join(output_folder, f"businessunit_{safe_bu_name}.csv")
    subset.to_csv(output_path, index=False)

    print(f"Kaydedildi: {output_path} ({len(subset)} satır)")

print("✅ Tüm veriler businessunit'e göre başarıyla bölündü!")"""

# Eğer 'time_stamp' zamanı da dahil etmek istersen listeye ekleyebilirsin ama normalde zamana gerek yok.

# Fonksiyonu çağır:
#plot_gender_distribution_by_feature(train_df, important_features)
# Kullanım
#count_total_rows(folder_path)
