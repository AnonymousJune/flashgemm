#include <stdio.h>
#include <sys/time.h>
#include <math.h>

static double gtod_ref_time_sec = 0.0;

int Check_result(float *C, float *C1, long M, long N)
{

	int i, j, flag = 0;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			if (fabs(C[i * N + j] - C1[i * N + j]) > 1.0e-3)
			{
				printf("i = %-10d, j= %-10d\n", i, j);
				printf("C= %-10.3lf , C1= %-10.3lf\n", C[i * N + j], C1[i * N + j]);
				printf("结果错误!\n");
				return 0;
			}
		}
	}
	// printf("结果正确\n");
	return 1;
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

void random_matrix(int m, int n, float *a)
{
	double drand48();
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i * n + j] = (float)drand48() - 0.5;
}
