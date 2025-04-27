import torch
import torch.nn as nn

# -------------------------------
# Attention付きLSTMモデルの定義（Attentionの有無を制御）
# -------------------------------
class LSTMWithOptionalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_attention, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        if self.use_attention:
            self.attention = Attention(hidden_dim)
            self.fc = nn.Linear(hidden_dim * 2, output_dim) # LSTM出力とAttention出力を結合
        else:
            self.fc = nn.Linear(hidden_dim, output_dim) # LSTM出力のみを使用

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_dropped = self.dropout(lstm_out) # LSTM出力にドロップアウト適用
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out_dropped)
            combined = torch.cat((lstm_out_dropped[:, -1, :], context_vector[:, -1, :]), dim=1)
            out = self.fc(self.dropout(combined)) # 結合後の特徴量にドロップアウト適用
            return out, attention_weights
        else:
            out = self.fc(lstm_out_dropped[:, -1, :]) # LSTM出力にドロップアウト適用
            return out, None # Attentionを使わない場合はNoneを返す
        
# -------------------------------
# Attention付きLSTMモデルの定義
# -------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        attn_scores = torch.bmm(lstm_output, self.attn_proj(lstm_output).transpose(1, 2))
        # attn_scores: (batch_size, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_weights: (batch_size, seq_len, seq_len)
        context_vector = torch.bmm(attn_weights, lstm_output)
        # context_vector: (batch_size, seq_len, hidden_dim)
        return context_vector, attn_weights