class CSRMatrix:
    def __init__(self, n):
        # val数组存储非零元素
        val = []
        # col_ind数组存储列索引
        col_ind = []
        # row_ptr数组存储行指针
        row_ptr = [0]  # 第一个元素总是0

        for i in range(n):
            # 主对角线元素
            val.append(2)
            col_ind.append(i)
            
            if i > 0:  # 上对角线元素
                val.append(-1)
                col_ind.append(i - 1)
            
            if i < n - 1:  # 下对角线元素
                val.append(-1)
                col_ind.append(i + 1)
            
            # 更新行指针
            row_ptr.append(len(val))

        # 由于Python索引从0开始，而CSR格式索引从1开始，因此对索引进行+1操作
        self.val = val
        self.col_ind = [x + 1 for x in col_ind]  # 列索引从1开始
        self.row_ptr = [x + 1 for x in row_ptr]  # 行指针从1开始

    def __str__(self):
        return f"CSR Matrix: {{'val': {self.val}, 'col_ind': {self.col_ind}, 'row_ptr': {self.row_ptr}}}"


# 创建CSR矩阵实例
csr_matrix = CSRMatrix(5)
print(csr_matrix)
print(csr_matrix.val[csr_matrix.row_ptr[1]])