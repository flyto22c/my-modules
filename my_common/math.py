
def my_round(float_val, n):
    '''
    Description: 自定义四舍五入函数。
    Args:
        val_f:  浮点数，一定有小数点。  -123.00       '-123.4567000'
    	n: 保留小数位数，非负数。       n >= 0         n = 5
    Example:
       my_round(-123.4567000 , 3) : -123.457
    '''
    str_val = str(float_val) + '0' * n + '000'  # 右边补一些0
    index_point = str_val.index('.')
    index_cut_right = index_point + n + 1  # 切分点(index_cut_left|index_cut_right)(-123.45670|00)

    str_cut_left = str_val[:index_cut_right]
    char_cut_right = str_val[index_cut_right]
    float_cut_left = float(str_cut_left)  # -123.456

    if char_cut_right > '4':  # 进位操作
        if float_cut_left < 0:
            float_cut_left = float_cut_left - 10 ** (-n)
        else:
            float_cut_left = float_cut_left + 10 ** (-n)
    # float_cut_left = -123.45700000000001
    float_cut_left = float(str(float_cut_left)[:index_cut_right])
    return float_cut_left
