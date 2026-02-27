import pandas as pd
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]

def plot(df1,df2):
    # 平滑函数：移动平均
    def smooth(data, window_size=10):
        return data.rolling(window=window_size, min_periods=1).mean()


    # 平滑数据
    df1['Value_smooth'] = smooth(df1['Value'], window_size=15)
    df2['Value_smooth'] = smooth(df2['Value'], window_size=15)

    # 假设两张表都有 'step' 和 'value' 两列
    plt.figure(figsize=(10, 6))
    # Curve 1：原始 + 平滑
    plt.plot(df1['Step'], -df1['Value'], color='red', alpha=0.4, linewidth=2)
    plt.plot(df1['Step'], -df1['Value_smooth'], label='GE-MHA',color='red', linewidth=2)

    # Curve 2：原始 + 平滑
    plt.plot(df2['Step'], -df2['Value'], color='orange', alpha=0.4, linewidth=2)
    plt.plot(df2['Step'], -df2['Value_smooth'],label='MHA', color='orange', linewidth=2)

    plt.ylim(-7.5,-4.2)
    # 美化图表
    plt.xlabel('Step',fontsize=20)
    plt.ylabel('Reward',fontsize=20)
    # plt.title('Comparison of Two Curves')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)

    # 保存或显示
    plt.savefig('rewardstep50.png', dpi=300)
    plt.show()

# 读取两个 CSV 文件
df1 = [pd.read_csv('hecvrp_rollout_50_6_reward.csv')]  # 替换为你的文件路径
df2 = [pd.read_csv('hecvrp_rollout_50_6_am_reward.csv')]
for d1,d2 in zip(df1,df2):
    plot(d1,d2)