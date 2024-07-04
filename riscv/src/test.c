#include <stdio.h>
// 引入 cblas 头文件
#include <cblas.h>

int main(void) {
    // 定义两个向量 x 和 y
    const double x[5] = {1.0, 2.0, -3.0, 4.0, 5.0};
    const double y[5] = {9.0, 7.0, 5.0, -3.0, 1.0};
    // 定义向量的长度和步长
    const int n = 5;
    const int incx = 1;
    const int incy = 1;
    // 定义点积结果变量
    double dotu;
    // 调用 cblas_ddot 函数计算点积
    dotu = cblas_ddot(n, x, incx, y, incy);
    // 打印结果
    printf("The dot product of x and y is %f\n", dotu);

    return 0;
}
