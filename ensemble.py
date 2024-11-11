import argparse
import numpy as np
import pygad
from tqdm import tqdm

# 定义日志文件名称
log_filename = "genetic_algorithm.log"

# 定义日志文件开头部分的函数，记录npy文件列表
def log_npy_files(files):
    with open(log_filename, 'a') as log_file:
        log_file.write('\n')
        log_file.write("Loaded npy files:\n")
        for file in files:
            log_file.write(f"{file}\n")
        log_file.write("\n")

# 计算集成后的准确率，用于遗传算法的适应度函数
def fitness_func(ga_instance, solution, solution_idx):
    # 对结果文件进行加权求和，使用给定的 solution 权重
    weighted_sum = sum(results[j] * solution[j] for j in range(len(results)))

    # 计算 Top-1 准确率
    top1_predictions = np.argmax(weighted_sum, axis=1)
    top1_accuracy = np.mean(top1_predictions == labels)

    return top1_accuracy  # 遗传算法最大化这个准确率

# 定义在每一代结束时记录当前最佳适应度值和权重
def on_generation(ga_instance):
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Generation: {ga_instance.generations_completed}, Best Fitness: {ga_instance.best_solution()[1]}")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Generation {ga_instance.generations_completed} - Best Fitness: {best_solution_fitness}\n")
        log_file.write("Weights: " + ", ".join(f"{w:.6f}" for w in best_solution) + "\n")

# 主优化函数，使用遗传算法找到最优权重
def optimize_weights_ga(results, labels, num_generations=100, num_parents_mating=5):
    num_files = len(results)

    # 使用 PyGAD 定义遗传算法实例
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=10,  # 每代的种群大小
        num_genes=num_files,  # 每个 solution 的基因数量等于文件数量
        init_range_low=0.0,
        init_range_high=1.0,
        mutation_percent_genes=20,  # 每代变异的基因百分比
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        gene_type=float,
        keep_parents=2,
        on_generation=on_generation
    )

    # 开始遗传算法优化
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    optimized_weights = best_solution / np.sum(best_solution)  # 归一化权重
    print("Optimized weights:", optimized_weights)

    return optimized_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', help='List of npy files containing evaluation results', required=True)
    parser.add_argument('--alpha', nargs='+', type=float, help='Manually specified weights for each npy file')
    args = parser.parse_args()

    # 加载真实标签
    labels = np.load('./data/val_label.npy', mmap_mode='r')
    print("Label shape:", labels.shape)

    # 加载所有结果文件
    results = [np.load(file) for file in args.files]

    # 在日志文件中记录加载的 npy 文件
    log_npy_files(args.files)

    # 检查是否手动指定了 alpha 权重
    if args.alpha:
        if len(args.alpha) != len(results):
            raise ValueError("The number of alpha weights must match the number of npy files.")
        optimized_weights = np.array(args.alpha) / np.sum(args.alpha)  # 归一化手动指定的 alpha 权重
        print("Using manually specified alpha weights:", optimized_weights)
    else:
        # 使用遗传算法优化权重
        optimized_weights = optimize_weights_ga(results, labels, num_generations=200)

    # 计算优化后权重的集成模型准确率
    right_num = total_num = right_num_5 = 0
    final_ensemble_result = []

    for i in tqdm(range(len(labels))):
        label = labels[i]

        # 使用优化权重对结果加权求和
        r = sum(results[j][i] * optimized_weights[j] for j in range(len(results)))
        final_ensemble_result.append(r)  # 保存每个样本的加权结果

        # 计算 Top-5 准确率
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(label) in rank_5)

        # 计算 Top-1 准确率
        r = np.argmax(r)
        right_num += int(r == int(label))
        total_num += 1

    # 输出最终准确率
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    # 保存最终的集成结果为 npy 文件，文件名包含 Top1 准确率
    final_ensemble_result = np.array(final_ensemble_result)
    np.save(f"ens_{acc:.4f}.npy", final_ensemble_result)
    print(f"Ensemble results saved to ens_{acc:.4f}.npy")

    # 在日志文件中记录最终的准确率和权重
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\nFinal Top1 Accuracy: {acc * 100:.4f}%\n")
        log_file.write(f"Final Top5 Accuracy: {acc5 * 100:.4f}%\n")
        log_file.write("Final Optimized Weights: " + ", ".join(f"{w:.4f}" for w in optimized_weights) + "\n")
    print(f"Log file saved to {log_filename}")
