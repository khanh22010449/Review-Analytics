import os
import pandas as pd

def split_data(df, val_size=0.15, test_size=0.2, random_state=42,
               save_csv=True, output_dir='.', prefix=''):
    """
    Chia DataFrame thành train/val/test theo tỉ lệ (tỉ lệ so với toàn bộ df).
    - val_size: tỉ lệ cho validation (ví dụ 0.15)
    - test_size: tỉ lệ cho test (ví dụ 0.2)
    - train_size = 1 - val_size - test_size
    - Nếu save_csv=True thì sẽ lưu train.csv, val.csv, test.csv tại output_dir (tên có thể có prefix)
    Trả về: (train_df, val_df, test_df)
    """
    if not (0 <= val_size < 1 and 0 <= test_size < 1):
        raise ValueError("val_size và test_size phải nằm trong [0,1).")
    if val_size + test_size >= 1.0:
        raise ValueError("Tổng val_size + test_size phải < 1.0.")
    
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    n = len(shuffled)
    n_test = int(round(test_size * n))
    n_val = int(round(val_size * n))
    n_train = n - n_val - n_test 
    
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    val = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_val:].reset_index(drop=True)
    
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(os.path.join(output_dir, f'{prefix}train.csv'), index=False)
        val.to_csv(os.path.join(output_dir, f'{prefix}val.csv'), index=False)
        test.to_csv(os.path.join(output_dir, f'{prefix}test.csv'), index=False)
    
    return train, val, test

if __name__ == "__main__":
    data = "train-problem.csv"
    df = pd.read_csv(data)
    split_data(df, val_size=0.15, test_size=0.2, random_state=42,
               save_csv=True, output_dir='data/', prefix='problem_')
