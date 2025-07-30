# plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_convergence(lora_losses, full_losses, save_path):
    """绘制并保存训练损失收敛曲线图。"""
    plt.figure(figsize=(10, 6))
    plt.plot(lora_losses, label='LoRA Fine-tuning Loss')
    plt.plot(full_losses, label='Full Fine-tuning Loss')
    plt.title('Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'convergence_curve.svg'),format='svg',dpi=300)
    plt.close()
    print("收敛曲线图已保存。")

def plot_bar_comparison(results, save_path):
    """绘制并保存参数量和评估指标的柱状对比图。"""
    labels = list(results.keys())
    params = [res['params'] for res in results.values()]
    bleus = [res['bleu'] for res in results.values()]
    rouges = [res['rouge'] for res in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    ax1.bar(x, params, width * 1.5, label='Trainable Parameters', color='skyblue')
    ax1.set_ylabel('Trainable Parameters (log scale)')
    ax1.set_title('Comparison of Trainable Parameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_yscale('log')
    ax1.legend()

    ax2.bar(x - width/2, bleus, width, label='BLEU Score', color='lightgreen')
    ax2.bar(x + width/2, rouges, width, label='ROUGE-L F1 Score', color='gold')
    ax2.set_ylabel('Score')
    ax2.set_title('Comparison of Evaluation Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0, max(max(bleus), max(rouges)) * 1.2 if any(bleus) or any(rouges) else 1)
    ax2.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'bar_chart_comparison.svg'),format='svg',dpi=300)
    plt.close()
    print("柱状对比图已保存。")

def plot_radar_chart(results, save_path):
    """绘制并保存综合性能雷达图。"""
    labels = list(results.keys())
    params = [res['params'] for res in results.values()]
    bleus = [res['bleu'] for res in results.values()]
    rouges = [res['rouge'] for res in results.values()]

    def param_efficiency_score(p):
        if p <= 1: return 0
        return 1 / np.log10(p)

    param_scores = [param_efficiency_score(p) for p in params]
    max_param_score = max(param_scores) if param_scores else 1
    if max_param_score == 0: max_param_score = 1
    normalized_param_scores = [ps / max_param_score for ps in param_scores]
    
    metrics_for_radar = {
        'Param Efficiency (1/log(p))': normalized_param_scores,
        'BLEU Score': bleus,
        'ROUGE-L Score': rouges,
    }
    
    radar_labels = list(metrics_for_radar.keys())
    num_vars = len(radar_labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, label in enumerate(labels):
        values = [metrics_for_radar[key][i] for key in radar_labels]
        values += values[:1]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi): label.set_horizontalalignment('center')
        elif 0 < angle < np.pi: label.set_horizontalalignment('left')
        else: label.set_horizontalalignment('right')

    ax.set_ylim(0, 1.1)
    ax.set_title('Overall Performance Radar Chart (Larger Area is Better)', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'radar_chart_comparison.svg'),format='svg',dpi=300)
    plt.close()
    print("雷达对比图已保存。")