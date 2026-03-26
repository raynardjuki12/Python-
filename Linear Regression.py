import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# データ作成
# ============================================================
np.random.seed(42)
X = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
              70, 75, 80, 85, 90, 95, 100, 110, 120, 130]).reshape(-1, 1)
y = 5 * X.flatten() + 200 + np.random.normal(0, 30, len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# モデル学習
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

# 回帰直線用データ
X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_line = model.predict(X_line)

# ============================================================
# グラフ描画（2つのサブプロット）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#F8F9FA')

# ---------- 左：散布図 + 回帰直線 ----------
ax1 = axes[0]
ax1.set_facecolor('#FFFFFF')

# 学習データ
ax1.scatter(X_train, y_train, color='#4A90D9', s=80, zorder=5,
            edgecolors='white', linewidth=1.2, label='Training data')
# テストデータ
ax1.scatter(X_test, y_test, color='#E74C3C', s=100, zorder=5,
            marker='D', edgecolors='white', linewidth=1.2, label='Test data')
# 回帰直線
ax1.plot(X_line, y_line, color='#2ECC71', linewidth=2.5,
         label=f'Regression line\ny = {model.coef_[0]:.2f}x + {model.intercept_:.1f}')

# テスト予測点と残差
for xi, yi, ypi in zip(X_test.flatten(), y_test, y_pred_test):
    ax1.plot([xi, xi], [yi, ypi], color='#E74C3C', linewidth=1,
             linestyle='--', alpha=0.5, zorder=4)

ax1.set_title('Scatter Plot & Regression Line', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Area (m²)', fontsize=12)
ax1.set_ylabel('Price (10,000 yen)', fontsize=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 情報テキスト
info = (f"Coefficient : {model.coef_[0]:.4f}\n"
        f"Intercept   : {model.intercept_:.4f}\n"
        f"R² (test)   : {r2:.4f}\n"
        f"R² (train)  : {model.score(X_train, y_train):.4f}")
ax1.text(0.97, 0.05, info, transform=ax1.transAxes,
         fontsize=9.5, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#EAF4FB', edgecolor='#AED6F1', alpha=0.9),
         family='monospace')

# ---------- 右：実測値 vs 予測値 ----------
ax2 = axes[1]
ax2.set_facecolor('#FFFFFF')

all_pred = model.predict(X)
ax2.scatter(y, all_pred, color='#9B59B6', s=80, edgecolors='white',
            linewidth=1.2, zorder=5, label='Actual vs Predicted')

# 理想直線（y = x）
val_min = min(y.min(), all_pred.min()) - 20
val_max = max(y.max(), all_pred.max()) + 20
ax2.plot([val_min, val_max], [val_min, val_max], color='#E67E22',
         linewidth=2, linestyle='--', label='Ideal (y = x)')

ax2.set_title('Actual vs Predicted', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Actual Price (10,000 yen)', fontsize=12)
ax2.set_ylabel('Predicted Price (10,000 yen)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(val_min, val_max)
ax2.set_ylim(val_min, val_max)

ax2.text(0.97, 0.05, f"R² (all data) : {r2_score(y, all_pred):.4f}",
         transform=ax2.transAxes, fontsize=9.5,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5EEF8', edgecolor='#D2B4DE', alpha=0.9),
         family='monospace')

# ---------- 全体タイトル ----------
fig.suptitle('Linear Regression Analysis  (sklearn.linear_model.LinearRegression)',
             fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved!")