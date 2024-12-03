import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

# 定義神經網路模型函數
def create_nn_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
    return model

# 使用當前工作目錄作為資料夾路徑
base_dir = os.getcwd()

# 遍歷所有資料夾
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.isdir(folder_path):
        # 讀取資料
        X_train = pd.read_csv(os.path.join(folder_path, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(folder_path, 'y_train.csv')).values.ravel()
        X_test = pd.read_csv(os.path.join(folder_path, 'X_test.csv'))

        # 資料平衡
        df = pd.DataFrame(X_train)
        df['y'] = y_train

        df_majority = df[df.y == 0]
        df_minority = df[df.y == 1]

        df_minority_upsampled = resample(df_minority, 
                                          replace=True,     
                                          n_samples=len(df_majority),    
                                          random_state=123)

        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        X_train = df_upsampled.drop('y', axis=1)
        y_train = df_upsampled['y']

        # 特徵標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 分割資料
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # 訓練 XGBoost 模型
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train_split, y_train_split)

        # 訓練神經網路模型
        nn_model = create_nn_model((X_train_split.shape[1],))
        nn_model.fit(X_train_split, y_train_split, epochs=150, batch_size=32, validation_split=0.2, 
                     callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

        # 預測驗證集
        y_val_pred_nn = nn_model.predict(X_val).flatten()
        y_val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]

        # 計算 AUC
        auc_score = roc_auc_score(y_val, (y_val_pred_nn*0.5 + y_val_pred_xgb*0.5) )
        print(f"AUC Score: {auc_score}")

        # 預測測試集
        y_pred_proba_nn = nn_model.predict(X_test).flatten()
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # 使用平均的方式進行集成
        y_pred_proba = (y_pred_proba_nn*0.5 + y_pred_proba_xgb*0.5)
        
        # 儲存預測結果
        y_predict_path = os.path.join(folder_path, 'y_predict.csv')
        pd.DataFrame(y_pred_proba, columns=['y_predict']).to_csv(y_predict_path, index=False)

        print(f"已處理資料夾: {folder_path}，預測結果已寫入 {y_predict_path}")
