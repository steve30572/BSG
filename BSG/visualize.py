import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
model_names = ['Model 1', 'Model 2', 'My Model', 'Model 4', 'Model 5']
model_values = [0.34, 0.98, 0.79, 0.96, 0.88]

# 기준 범위
lower_bound = 0.76
upper_bound = 0.91

# 색상을 조건에 따라 설정 (기준 범위 내에 있으면 초록색, 아니면 빨간색)
colors = ['red' if value < lower_bound or value > upper_bound else 'green' for value in model_values]

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, model_values, color=colors)

# 기준 범위 시각적으로 강조 (회색 영역)
plt.axhspan(lower_bound, upper_bound, color='gray', alpha=0.3, label=f"Range {lower_bound}-{upper_bound}")

# 그래프 꾸미기
plt.xlabel('Models')
plt.ylabel('Values')
plt.title('Comparison of Model Values with Desired Range (0.76-0.91)')
plt.ylim(0, 1)
plt.legend()

# 각 막대에 값 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.savefig('./visualize_intro.png')
# plt.show()
