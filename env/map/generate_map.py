import numpy as np
import pandas as pd
import random


def generate_blocked_ones_matrix(rows, cols, zero_ratio, block_count):
    """
    生成一个矩阵，矩阵中包含指定比例的0和1，并且1是分散在多个连续的块中。

    参数:
        rows (int): 矩阵的行数
        cols (int): 矩阵的列数
        zero_ratio (float): 0的比例 (0到1之间)
        block_count (int): 要生成的1的连续块数量

    返回:
        matrix (ndarray): 生成的矩阵
    """
    # 确定矩阵中0和1的数量
    num_elements = rows * cols
    num_zeros = int(num_elements * zero_ratio)
    num_ones = num_elements - num_zeros

    # 初始化矩阵为0
    matrix = np.zeros((rows, cols), dtype=int)

    # 计算每块的平均大小
    ones_per_block = num_ones // block_count

    # 随机生成每个块的起始位置和方向
    for _ in range(block_count):
        block_placed = False
        while not block_placed:
            # 随机选择块的起始点
            start_row = random.randint(0, rows - 1)
            start_col = random.randint(0, cols - 1)

            # 随机选择方向 (水平或垂直)
            direction = random.choice(['horizontal', 'vertical'])

            # 检查是否有足够的空间放置块
            if direction == 'horizontal' and start_col + ones_per_block <= cols:
                if np.all(matrix[start_row, start_col:start_col + ones_per_block] == 0):
                    matrix[start_row, start_col:start_col + ones_per_block] = 1
                    block_placed = True
            elif direction == 'vertical' and start_row + ones_per_block <= rows:
                if np.all(matrix[start_row:start_row + ones_per_block, start_col] == 0):
                    matrix[start_row:start_row + ones_per_block, start_col] = 1
                    block_placed = True

    return matrix


def save_matrix_to_excel(matrix, rows, cols, zero_ratio, file_name):
    """
    将矩阵保存到Excel文件中，文件名包含行数、列数、0的比例和块数量。

    参数:
        matrix (ndarray): 要保存的矩阵
        rows (int): 矩阵的行数
        cols (int): 矩阵的列数
        zero_ratio (float): 0的比例 (0到1之间)
        block_count (int): 1的连续块数量
    """
    filename = file_name
    df = pd.DataFrame(matrix)
    df.to_excel(filename, index=False, header=False)
    print(f"矩阵已保存到 {filename}")


# 示例：生成一个10x10的矩阵，30%为0，70%为分成5个连续块的1，并保存到Excel文件
rows = int(10)        # 行数
cols = int(10)        # 列数
zero_ratio = float(0.3)  # 0 的比例
block_count = int(5)     # 块的数量
ratio = 1 - zero_ratio

file_name = f"train_map_{rows}x{cols}_o{ratio}.xlsx"
matrix = generate_blocked_ones_matrix(rows, cols, ratio, file_name)
save_matrix_to_excel(matrix, rows, cols, ratio, block_count)






