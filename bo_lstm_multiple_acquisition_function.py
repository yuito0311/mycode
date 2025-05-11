import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from scipy.stats import norm
import matplotlib.pyplot as plt

# LSTMの呼び出し
from LSTMwithAttention import LSTMWithOptionalAttention, Attention

def bo_lstm_hyperparams(dataset, num_epochs, bo_iteration_number=15, display_flag=False, bo_iteration_plot=False):
    """
    LSTMモデルのハイパーパラメータを、ベイズ最適化を用いてチューニングする関数

    Args:
        dataset (pd.DataFrame): 入力用の時系列データセット。1列目が目的変数、それ以降が説明変数での入力を想定している。
        num_epochs (int): LSTMモデルにおけるエポック数。
        bo_iteration_number (int): ベイズ最適化の試行回数。デフォルトは15。
        display_flag (bool): ベイズ最適化の途中経過を表示するか。デフォルトはFalse。
        bo_iteration_flag (bool): ベイズ最適化の過程をグラフ表示するか。デフォルトはFalse。

    Return:
        tuple: 最適化されたLSTM、窓枠サイズ, バッチサイズ、学習率のタプル。
    """

    # 固定のパラメータ
    # 入力次元
    input_dim = dataset.shape[1] - 1

    # 出力次元
    output_dim = 1

    # ハイパーパラメータの探索候補
    seq_length = [10, 50, 100, 200]  # sliding_windowのサイズ
    hidden_dim = [2, 4, 8, 16, 32]  # 隠れ層の数(小さめに設定)
    batch_size = [4, 8, 16, 32]  # バッチサイズ
    lr = [1e-5, 1e-4, 1e-3, 1e-2]  # 学習率
    dropout_rate = [0.2, 0.3, 0.4, 0.5]  # ドロップアウト率
    use_attention = [True, False]   # Attention層を使うかどうかを選択 (True or False)

    # 実験計画法の条件
    doe_number_of_selecting_samples = 15   # 選択するサンプル数
    doe_number_of_random_searches = 100   # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数

    # BOの設定
    bo_iterations = np.arange(0, bo_iteration_number + 1)
    bo_gp_fold_number = 5  # BOのGPを構築するためのcvfold数
    bo_number_of_selecting_samples = 1   # 選択するサンプル数
    # bo_regression_method = 'gpr_kernels'  # gpr_one_kernel', 'gpr_kernels'
    bo_regression_method = 'gpr_one_kernel'  # gpr_one_kernel', 'gpr_kernels'
    bo_kernel_number = 2   # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    #acquisition_functions = ['PTR', 'PI', 'EI', 'MI']
    acquisition_functions = ['PI', 'EI']
    target_range = [1, 100]   # PTR
    relaxation = 0.01  # EI, PI
    delta = 10 ** -6  # MI

    # ハイパーパラメータの探索空間の作成
    parameter_candidates = []
    for window_size in seq_length:
        for hidden in hidden_dim:
            for batch in batch_size:
                for learning_rate in lr:
                    for attention in use_attention:
                        for drop in dropout_rate:
                            parameter_candidates.append([window_size, hidden, batch, learning_rate,
                                                        attention, drop])

    all_candidate_combinations_df =  pd.DataFrame(parameter_candidates)
    clm_name = ['window_size', 'hidden_dim', 'batch_size', 'learning_rate',
                'attention', 'dropout_rate']

    all_candidate_combinations_df.columns = clm_name


    numerical_variable_numbers = np.array([0, 1, 2, 3, 5])
    category_variable_numbers = np.array([4])
    category_columns = all_candidate_combinations_df.columns[category_variable_numbers]
    # ワンホット変換
    numerical_x = all_candidate_combinations_df.iloc[:, numerical_variable_numbers]
    category_x = all_candidate_combinations_df.iloc[:, category_variable_numbers].astype(int)
    # dummy_x = pd.get_dummies(category_x, columns=category_columns).astype(int)
    params_df = pd.concat([numerical_x, category_x], axis=1)

    #########################ここからベイズ最適化#########################

    # 結果の保存用リスト
    all_r2_scores = []
    bo_r2_scores = []
    trial_numbers = []


    # ベイズ最適化の繰り返し
    for bo_iter in bo_iterations:
        if display_flag:
            print(f'Bayesian optimization iteration : {bo_iter + 1} / {bo_iteration_number}')
    #   print('='*10)
        if bo_iter == 0:  # 最初の試行ではD最適基準を計算
            # D最適基準の計算
            autoscaled_params_df = (params_df - params_df.mean(axis=0)) / params_df.std(axis=0, ddof=1)  # 計算のために標準化
            all_indexes = list(range(autoscaled_params_df.shape[0]))  # indexを取得

            np.random.seed(11)  # 乱数を生成するためのシードを固定
            for random_search_number in range(doe_number_of_random_searches):
                # 1. ランダムに候補を選択
                new_selected_indexes = np.random.choice(all_indexes, doe_number_of_selecting_samples, replace=False)
                new_selected_samples = autoscaled_params_df.iloc[new_selected_indexes, :]
                # 2. D 最適基準を計算
                xt_x = np.dot(new_selected_samples.T, new_selected_samples)
                d_optimal_value = np.linalg.det(xt_x)
                # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
                if random_search_number == 0:
                    best_d_optimal_value = d_optimal_value.copy()
                    selected_sample_indexes = new_selected_indexes.copy()
                else:
                    if best_d_optimal_value < d_optimal_value:
                        best_d_optimal_value = d_optimal_value.copy()
                        selected_sample_indexes = new_selected_indexes.copy()
            selected_sample_indexes = list(selected_sample_indexes)  # リスト型に変換

            # 選択されたサンプル、選択されなかったサンプル
            selected_params_df = params_df.iloc[selected_sample_indexes, :]  # 選択されたサンプル
            true_selected_params_df = all_candidate_combinations_df.iloc[selected_sample_indexes, :]
            bo_params_df = selected_params_df.copy()  # BOのGPモデル構築用データを作成
            remaining_indexes = np.delete(all_indexes, selected_sample_indexes)  # 選択されなかったサンプルのインデックス
            remaining_params_df = params_df.iloc[remaining_indexes, :]  # 選択されなかったサンプル
            true_remaining_params_df = all_candidate_combinations_df.iloc[remaining_indexes, :]

            # 選択された全候補でGPRの計算
            params_with_score_df = params_df.copy()  # cvのscoreが含まれるdataframe
            params_with_score_df['score'] = np.nan  # 初期値はnanを設定

        else:  # 2回目以降では前回の結果をもとにする
            selected_sample_indexes = next_samples_df.index  # 提案サンプルのindex
            selected_params_df = params_df.loc[selected_sample_indexes, :]  # 次に計算するサンプル
            true_selected_params_df = all_candidate_combinations_df.loc[selected_sample_indexes, :]  # 次に計算するサンプル
            bo_params_df = pd.concat([bo_params_df, selected_params_df], axis=0)  # BOのGPモデル構築用データは前回のデータと提案サンプルをマージする
            remaining_params_df = params_df.loc[params_with_score_df['score'].isna(), :]  # 選択されなかったサンプル
            remaining_params_df = remaining_params_df.drop(index=selected_sample_indexes)
            true_remaining_params_df = all_candidate_combinations_df.loc[params_with_score_df['score'].isna(), :]  # 選択されなかったサンプル
            true_remaining_params_df = true_remaining_params_df.drop(index=selected_sample_indexes)

        # 選ばれたサンプル（パラメータの組み合わせ）を一つずつ計算する
        for i_n, selected_params_idx in enumerate(selected_sample_indexes):
            selected_params = true_selected_params_df.loc[selected_params_idx, :]  # サンプルの選択

            # データ拡張の選択
            selected_seq_length = selected_params['window_size']
            selected_hidden_dim = selected_params['hidden_dim']
            selected_batch_size = selected_params['batch_size']
            selected_lr = selected_params['learning_rate']
            selected_attention = selected_params['attention']
            selected_dropout_late = selected_params['dropout_rate']

            data = dataset.values.astype('float32')

            inputs = data[:, 1:]
            targets = data[:, 0]

            # window_sizeごとにデータを区切ります
            input_sequences, target_sequences = create_sequences(inputs, targets, selected_seq_length)

            # 訓練用データと検証用データに分けます
            X_train, X_val, y_train, y_val = train_test_split(input_sequences, target_sequences, test_size=0.3, shuffle=False)

            # ★ 訓練データの平均と標準偏差を計算
            train_mean = np.mean(X_train, axis=(0, 1))  # 時間ステップと特徴量の平均
            train_std = np.std(X_train, axis=(0, 1))    # 時間ステップと特徴量の標準偏差

            # ★ 訓練データと検証データを、訓練データの平均と標準偏差でスケーリング
            X_train_scaled = (X_train - train_mean) / train_std
            X_val_scaled = (X_val - train_mean) / train_std

            y_train_scaled = (y_train - np.mean(y_train)) / np.std(y_train)
            y_val_scaled = (y_val - np.mean(y_train)) / np.std(y_train) # 訓練データの平均と標準偏差を使用

            # ★ スケールされたデータでTensorを作成
            train_inputs_tensor = torch.tensor(X_train_scaled).float()
            val_inputs_tensor = torch.tensor(X_val_scaled).float()

            train_targets_tensor = torch.tensor(y_train_scaled).float().unsqueeze(1)
            val_targets_tensor = torch.tensor(y_val_scaled).float().unsqueeze(1)

            train_dataset_tensor = TensorDataset(train_inputs_tensor, train_targets_tensor)
            val_dataset_tensor = TensorDataset(val_inputs_tensor, val_targets_tensor)

            train_loader = DataLoader(train_dataset_tensor, batch_size=int(selected_batch_size), shuffle=False)
            val_loader = DataLoader(val_dataset_tensor, batch_size=int(selected_batch_size), shuffle=False)

            model = LSTMWithOptionalAttention(input_dim, int(selected_hidden_dim), output_dim,
                                                selected_attention, selected_dropout_late)

            # 学習準備
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=selected_lr)
            train_losses = []

            # 訓練用コード
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                all_train_predictions = []
                all_true_train_targets = []

                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)

                    # 訓練データの予測値を保存
                    all_train_predictions.extend(outputs.detach().cpu().numpy().flatten())
                    all_true_train_targets.extend(targets.detach().cpu().numpy().flatten())

                avg_loss = epoch_loss / len(train_dataset_tensor)
                train_losses.append(avg_loss)

            # 検証用
            model.eval()
            val_loss = 0
            val_r2_scores = []
            all_val_predictions = []
            all_true_val_targets = []
            all_attention_weights = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs, attention_weights = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                    all_val_predictions.extend(outputs.cpu().numpy().flatten())
                    all_true_val_targets.extend(targets.cpu().numpy().flatten())
                    if model.use_attention:  # モデルのインスタンス変数use_attentionを参照
                        if attention_weights is not None:  # attention_weightsがNoneでないことを確認
                            all_attention_weights.extend(attention_weights.cpu().numpy())

                average_test_loss = val_loss / len(val_dataset_tensor)
                # ★ 検証データ全体のR2スコアを計算
                # スケールを逆変換
                y_train_mean = np.mean(y_train)
                y_train_std = np.std(y_train)

                all_val_predictions_original_scale = (np.array(all_val_predictions) * y_train_std) + y_train_mean
                all_true_val_targets_original_scale = (np.array(all_true_val_targets) * y_train_std) + y_train_mean
                #val_r2 = r2_score(all_true_val_targets_original_scale, all_val_predictions_original_scale)
                val_r2 = r2lm(all_true_val_targets_original_scale, all_val_predictions_original_scale)
                val_r2_scores.append(val_r2)

            current_r2 = val_r2_scores[-1]
            params_with_score_df.loc[selected_params_idx, 'score'] = current_r2  # データの保存
            all_r2_scores.append(current_r2)
            trial_numbers.append(bo_iter if bo_iter > 0 else f'D-Opt {i_n+1}')

        if display_flag:
            print('Best score :', params_with_score_df['score'].max())
            print('='*10)

        # 最後はBOの計算をしないためbreak
        if bo_iter + 1 == bo_iteration_number:
            break

        # Bayesian optimization
        bo_x_data = bo_params_df.copy()  # GP学習用データはGMRの結果があるサンプル
        bo_x_prediction = remaining_params_df.copy()  # predictionは選択されていない（GMRの結果がない）サンプル
        bo_y_data = params_with_score_df.loc[bo_params_df.index, 'score']  # yはGMRのr2cv

        # カーネル 11 種類
        bo_kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
                      ConstantKernel() * RBF() + WhiteKernel(),
                      ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
                      ConstantKernel() * RBF(np.ones(bo_x_data.shape[1])) + WhiteKernel(),
                      ConstantKernel() * RBF(np.ones(bo_x_data.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
                      ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                      ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                      ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
                      ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                      ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                      ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]

        next_samples = pd.DataFrame([], columns=selected_params_df.columns)  # 次のサンプルを入れる変数を準備

        # 次の候補を複数提案する繰り返し工程
        for bo_sample_number in range(bo_number_of_selecting_samples):
            # オートスケーリング
            bo_x_data_std = bo_x_data.std()
            bo_x_data_std[bo_x_data_std == 0] = 1
            autoscaled_bo_y_data = (bo_y_data - bo_y_data.mean()) / bo_y_data.std()
            autoscaled_bo_x_data = (bo_x_data - bo_x_data.mean()) / bo_x_data_std
            autoscaled_bo_x_prediction = (bo_x_prediction - bo_x_data.mean()) / bo_x_data_std

            # モデル構築
            if bo_regression_method == 'gpr_one_kernel':
                bo_selected_kernel = bo_kernels[bo_kernel_number]
                bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_selected_kernel)

            elif bo_regression_method == 'gpr_kernels':
                # クロスバリデーションによるカーネル関数の最適化
                bo_cross_validation = KFold(n_splits=bo_gp_fold_number, random_state=9, shuffle=True)  # クロスバリデーションの分割の設定
                bo_r2cvs = []  # 空の list。カーネル関数ごとに、クロスバリデーション後の r2 を入れていきます
                for index, bo_kernel in enumerate(bo_kernels):
                    bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_kernel)
                    estimated_bo_y_in_cv = np.ndarray.flatten(cross_val_predict(bo_model, autoscaled_bo_x_data, autoscaled_bo_y_data, cv=bo_cross_validation))
                    estimated_bo_y_in_cv = estimated_bo_y_in_cv * bo_y_data.std(ddof=1) + bo_y_data.mean()
                    bo_r2cvs.append(r2_score(bo_y_data, estimated_bo_y_in_cv))
                optimal_bo_kernel_number = np.where(bo_r2cvs == np.max(bo_r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
                optimal_bo_kernel = bo_kernels[optimal_bo_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数

                # モデル構築
                bo_model = GaussianProcessRegressor(alpha=0, kernel=optimal_bo_kernel, random_state=9)  # GPR モデルの宣言

            bo_model.fit(autoscaled_bo_x_data, autoscaled_bo_y_data)  # モデルの学習

            # 予測
            estimated_bo_y_prediction, estimated_bo_y_prediction_std = bo_model.predict(autoscaled_bo_x_prediction, return_std=True)
            estimated_bo_y_prediction = estimated_bo_y_prediction * bo_y_data.std() + bo_y_data.mean()
            estimated_bo_y_prediction_std = estimated_bo_y_prediction_std * bo_y_data.std()

            cumulative_variance = np.zeros(bo_x_prediction.shape[0])

            # 獲得関数の決定(今回のプログラムでは獲得関数を試行ごとに変更して使用します)
            selected_aquisition_function_number  = bo_iter % len(acquisition_functions)
            acquisition_function = acquisition_functions[selected_aquisition_function_number]

            # 獲得関数の計算
            if acquisition_function == 'MI':
                acquisition_function_prediction = estimated_bo_y_prediction + np.log(2 / delta) ** 0.5 * (
                        (estimated_bo_y_prediction_std ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
                cumulative_variance = cumulative_variance + estimated_bo_y_prediction_std ** 2
            elif acquisition_function == 'EI':
                acquisition_function_prediction = (estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) * \
                                                    norm.cdf((estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) /
                                                             estimated_bo_y_prediction_std) + \
                                                    estimated_bo_y_prediction_std * \
                                                    norm.pdf((estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) /
                                                             estimated_bo_y_prediction_std)
            elif acquisition_function == 'PI':
                acquisition_function_prediction = norm.cdf(
                        (estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) / estimated_bo_y_prediction_std)
            elif acquisition_function == 'PTR':
                acquisition_function_prediction = norm.cdf(target_range[1],
                                                            loc=estimated_bo_y_prediction,
                                                            scale=estimated_bo_y_prediction_std
                                                            ) - norm.cdf(target_range[0],
                                                                        loc=estimated_bo_y_prediction,
                                                                        scale=estimated_bo_y_prediction_std)
            acquisition_function_prediction[estimated_bo_y_prediction_std <= 0] = 0

            # 保存
            estimated_bo_y_prediction = pd.DataFrame(estimated_bo_y_prediction, bo_x_prediction.index, columns=['estimated_y'])
            estimated_bo_y_prediction_std = pd.DataFrame(estimated_bo_y_prediction_std, bo_x_prediction.index, columns=['std_of_estimated_y'])
            acquisition_function_prediction = pd.DataFrame(acquisition_function_prediction, index=bo_x_prediction.index, columns=['acquisition_function'])
    #       
            # 次のサンプル
            next_samples = pd.concat([next_samples, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)

            # x, y, x_prediction, cumulative_variance の更新
            bo_x_data = pd.concat([bo_x_data, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
            bo_y_data = pd.concat([bo_y_data, estimated_bo_y_prediction.loc[acquisition_function_prediction.idxmax()].iloc[0]], axis=0)
            bo_x_prediction = bo_x_prediction.drop(acquisition_function_prediction.idxmax(), axis=0)
            cumulative_variance = np.delete(cumulative_variance, np.where(acquisition_function_prediction.index == acquisition_function_prediction.iloc[:, 0].idxmax())[0][0])
        next_samples_df = next_samples.copy()

    # 結果の保存
# params_with_score_df.sort_values('score', ascending=False).to_csv('params_with_score.csv')
    # print(params_with_score_df)
    params_with_score_df_best = params_with_score_df.sort_values('score', ascending=False).iloc[0, :]  # r2が高い順にソート
    params_with_score_df.to_csv('params.csv')
    # best_r2

    optimal_window_size = int(params_with_score_df_best.iloc[0])  # 最適な窓サイズ
    optimal_hidden_dim = int(params_with_score_df_best.iloc[1])  # 最適な隠れ層サイズ
    optimal_batch_size = int(params_with_score_df_best.iloc[2])  # 最適なバッチサイズ
    optimal_learning_rate = params_with_score_df_best.iloc[3]  # 最適な学習率
    optimal_dropout_rate = params_with_score_df_best.iloc[4]  # 最適なドロップアウト率
    if int(params_with_score_df_best.iloc[5]) == 1:  # アテンションを適応させるかどうか
        optimal_attention = True
    else:
        optimal_attention = False

    model = LSTMWithOptionalAttention(input_dim, optimal_hidden_dim, output_dim,
                                        optimal_attention, optimal_dropout_rate)

    # ベイズ最適化の過程を描画します
    if bo_iteration_plot:
        plt.figure(figsize=(10, 6))

        initial_r2 = all_r2_scores[:doe_number_of_selecting_samples]
        bo_r2 = all_r2_scores[doe_number_of_selecting_samples -1:]
        initial_trials = np.arange(1, doe_number_of_selecting_samples + 1)
        # ベイズ最適化の試行回数をD最適化の最後に接続するように修正
        bo_trials = np.arange(doe_number_of_selecting_samples, len(all_r2_scores) +1)
        # D最適化によるサンプリングの表示
        plt.plot(initial_trials, initial_r2, marker='o', linestyle='-', color='blue', label='D-Optimal')
        # ベイズ最適化によるサンプリングの表示
        plt.plot(bo_trials, bo_r2, marker='o', linestyle='-', color='green', label='Bayesian Optimization')

        # D最適化とベイズ最適化との境界線を表示
        plt.axvline(x=doe_number_of_selecting_samples, color='red', linestyle='--', label='Start of Bayesian Optimization')

        plt.xlabel('Trial Number')
        plt.ylabel('R2 Score')
        plt.title('Bayesian Optimization Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('bo_progress.png')
        plt.show()


    # return optimal_window_size, optimal_hidden_dim, optimal_batch_size, optimal_learning_rate, optimal_dropout_rate, optimal_attention
    return model, optimal_window_size, optimal_batch_size, optimal_learning_rate

# Calculate r^2 based on the latest measured y-values
# measured_y and estimated_y must be vectors.
def r2lm(measured_y, estimated_y):
    measured_y = np.array(measured_y).flatten()
    estimated_y = np.array(estimated_y).flatten()
    return float(1 - sum((measured_y - estimated_y) ** 2) / sum((measured_y[1:] - measured_y[:-1]) ** 2))

def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(target[i+seq_length])
    return np.array(xs), np.array(ys)
