import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
# 配置与时间数据
configs = ['C10-RC3-V3', 'C10-RC3-V5', 'C20-RC3-V3', 'C20-RC3-V5', 'C50-RC6-V3', 'C50-RC6-V5']
mha_times = [2.53, 3.05, 5.22, 6.33, 11.08, 11.25]
ge_mha_times = [2.77, 3.61, 5.55, 6.38, 11.25, 13.64]

# 创建折线图
plt.figure(figsize=(8, 5))
plt.plot(configs, mha_times, label='MHA', color='orange', marker='o', linewidth=2, alpha=0.7)
plt.plot(configs, ge_mha_times, label='GE-MHA', color='red', marker='s', linewidth=2, alpha=0.7)

# 图形设置
plt.xlabel('配置',fontsize=20)
plt.ylabel('训练时间 (h)',fontsize=20)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig('trainingtime.png', dpi=300)
# 显示图像
plt.show()
