import matplotlib.pyplot as plt
import pandas as pd

bp_df = pd.read_csv('checkpoint/quant_noise_incre_bp_add_gaussian_noise/acc.csv').values[:, 1]
mfusion_df = pd.read_csv('checkpoint/quant_noise_incre_mfusion_add_gaussian_noise/acc.csv').values[:, 1]
bp_qn_df = pd.read_csv('checkpoint/quant_noise_incre_bp_qn_add_gaussian_noise/acc.csv').values[:, 1]
mfusion_qn_df = pd.read_csv('checkpoint/quant_noise_incre_mfusion_qn_add_gaussian_noise/acc.csv').values[:, 1]

plt.figure()
plt.plot(bp_df, label='bp')
plt.plot(mfusion_df, label='mfusion')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('checkpoint/bp_vs_mfusion_acc.png')
plt.show()
plt.close()

plt.figure()
plt.plot(bp_qn_df, label='bp_qn')
plt.plot(mfusion_qn_df, label='mfusion_qn')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('checkpoint/bp_qn_vs_mfusion_qn_acc.png')
plt.show()
plt.close()
