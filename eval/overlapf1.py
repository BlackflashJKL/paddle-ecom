def calculate_overlap_f1(string1, string2):
    # 计算最大连续公共子序列长度
    mcl_len = get_mcl_len(string1, string2)
    # 计算Overlap-F1指标
    # overlap_f1 = 2 * mcl_len / (len(string1) + len(string2))
    nume=2*mcl_len
    deno=len(string1)+len(string2)
    return nume,deno

def get_mcl_len(string1, string2):
    # 构建动态规划表
    m = [[0] * (len(string2) + 1) for _ in range(len(string1) + 1)]
    max_len = 0  # 记录最大连续公共子序列长度
    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i-1] == string2[j-1]:
                m[i][j] = m[i-1][j-1] + 1
                if m[i][j] > max_len:
                    max_len = m[i][j]
            else:
                m[i][j] = 0
    return max_len

# string1 = "终于我来上"
# string2 = "我来上海了"
# overlap_f1 = calculate_overlap_f1(string1, string2)
# print("Overlap-F1 for '{}' and '{}' is: {}".format(string1, string2, overlap_f1))