#include <sys/time.h>

static double gtod_ref_time_sec = 0.0;

float bf16_to_float(uint16_t bf16) {
    union {
        float f;
        uint32_t i;
    } u;
    // 提取符号位、指数位、尾数位
    uint32_t sign = (bf16 >> 15) & 0x1;
    uint32_t exponent = (bf16 >> 7) & 0xFF;
    uint32_t mantissa = bf16 & 0x7F;

    // 组合成FP32格式（指数偏移调整）
    u.i = (sign << 31) | (exponent << 23) | (mantissa << 16);
    return u.f;
}

uint16_t float_to_bf16(float value) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = value;
    // 提取符号位（1位）、指数位（8位）、尾数位（7位）
    uint32_t sign = (u.i >> 31) & 0x1;          // 符号位
    uint32_t exponent = (u.i >> 23) & 0xFF;     // FP32指数位（需调整为bf16的指数范围）
    uint32_t mantissa = (u.i >> 16) & 0x7F;     // FP32尾数高7位（bf16尾数）

    // 调整指数范围：FP32的指数偏移为127，bf16偏移为127，无需调整
    return (sign << 15) | (exponent << 7) | mantissa;
}

// 生成一个随机的BF16数
uint16_t generate_random_bf16() {
    float random_float = 2.0 * ((float)rand() / RAND_MAX) - 1.0; // -1 to 1
    uint16_t bf16_value = float_to_bf16(random_float);
    return bf16_value;
}

// 打印BF16
void print_bf16(uint16_t bf16_value) {
    // 为了可视化，这里将BF16还原为32位浮点数
    uint32_t float_value = bf16_value << 16;
    float *as_float = (float*)&float_value;
    printf("%-8.2f ", as_float);
}

// 打印BF16,16进制
void print_bf16_x(uint16_t bf16_value) {
    // 为了可视化，这里将BF16还原为32位浮点数
    uint32_t float_value = bf16_value << 16;
    float *as_float = (float*)&float_value;
    printf("%-6x", as_float);
}


void random_matrix_bf16(int m, int n, uint16_t *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = (uint16_t)generate_random_bf16();
}

void regular_matrix_bf16(int m, int n, uint16_t *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = float_to_bf16(i + j * 0.01);
}

void regular1_matrix_bf16(int m, int n, uint16_t *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = float_to_bf16(1.0);
}

void regular2_matrix_bf16(int m, int n, uint16_t *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = float_to_bf16(j);
}

void random_matrix_f32(int m, int n, float *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = 2.0 * ((float)rand() / RAND_MAX) - 1.0; // -1 to 1
}

void regular_matrix_f32(int m, int n, float *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = (i + 0.01 * j);
}

void regular1_matrix_f32(int m, int n, float *a)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = (1.0);
}

void show_matrix_fp32(long m, long n, float *a)
{
	long i, j;
	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++)
			printf("%-10.2f ", a[i * n + j]);
		printf("\n");
	}
		
}

void show_matrix_bf16(long m, long n, uint16_t *a)
{
	long i, j;
	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++)
			printf("%-8.4f ", bf16_to_float(a[i * n + j]));
		printf("\n");
	}
		
}

double dclock()
{
	double the_time, norm_sec;
	struct timeval tv;

	gettimeofday(&tv, NULL);

	if (gtod_ref_time_sec == 0.0)
		gtod_ref_time_sec = (double)tv.tv_sec;

	norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

	the_time = norm_sec + tv.tv_usec * 1.0e-6;

	return the_time;
}

int Check_result(float *C, float *C1, long M, long N)
{

  int i, j, flag = 0;

  for (i = 0; i < M; i++)
  {
    for (j = 0; j < N; j++)
    {
      if (abs(C[i * N + j] - C1[i * N + j]) > 1e-3)
      {
        printf("i = %-6d, j= %-6d\n", i, j);
        printf("C= %.3lf , C1= %.3lf\n", C[i * N + j], C1[i * N + j]);
        printf("结果错误!\n");
        return 0;
      }
    }
  }
  // printf("结果正确\n");
  return 1;
}

void f32_matrix_to_bf16_matrix(float *f32_matrix, uint16_t *bf16_matrix, long m, long n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            bf16_matrix[i * n + j] = float_to_bf16(f32_matrix[i * n + j]);
}

void random_matrix_int8(int m, int n, int8_t *a)
{
	// 设置随机数种子
	// srand(time(NULL));
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i * n + j] = (int8_t)(rand() % 256 - 128);
}

void random_matrix_uint8(long m, long n, uint8_t *a)
{
	long i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i * n + j] = (uint8_t)(rand() % 256);
}

void regular_matrix_int8(long m, long n, int8_t *a)
{
	long i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i * n + j] = (int8_t)(i+j);
}

void regular_matrix_uint8(long m, long n, uint8_t *a)
{
	long i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i * n + j] = (uint8_t)(i+j);
}

void show_matrix(long m, long n, int *a)
{
	long i, j;

	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++)
			printf("%-8d", a[i * n + j]);
		printf("\n");
	}
		
}

void show_matrix_int8(long m, long n, int8_t *a)
{
	long i, j;

	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++)
			printf("%-8d", a[i * n + j]);
		printf("\n");
	}
		
}
