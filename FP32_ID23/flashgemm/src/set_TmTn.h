#include <algorithm>
#include <vector>
#include <limits.h>
#include <math.h>
using namespace std;

int Tm=1, Tn=1;
vector<int> vec;

void Dete_grad_N_threads_nums(int T, long M, long N, int transa, int transb)
{
	int i;
	// Determines the number of threads to parallelize the N-dimension
	Tn = ceil(sqrt(T * N / M));
	for (i = 0; i < vec.size(); i++)
	{
		if (Tn <= vec[i])
		{
			Tn = vec[i];
			break;
		}
	}
	if (Tn >= T)
		Tn = T;
	else
	{
		if (transa == 0 && transb == 0 && (M / N) < 10)
		{
			if ((M / Tm) < (N / Tn))
			{
				Tn = vec[i + 1]; // 使M/Tm和N/Tn更相近
			}
		}
	}
	// Determines the number of threads to parallelize the M-dimension
	Tm = T / Tn;
}

void Dete_grad_M_threads_nums(int T, long M, long N, int transa, int transb)
{

	int i;
	// Determines the number of threads to parallelize the M-dimension
	Tm = ceil(sqrt(T * M / N));
	for (i = 0; i < vec.size(); i++)
	{
		if (Tm <= vec[i])
		{
			Tm = vec[i];
			break;
		}
	}
	if (Tm >= T)
		Tm = T;

	// Determines the number of threads to parallelize the N-dimension
	Tn = T / Tm;
}


void LibShalom_set_thread_nums(int num)
{
	int i;
	for (i = 1; i <= sqrt(num); i++)
	{
		if (num % i == 0)
		{
			vec.push_back(i);
			if (num != i * i)
				vec.push_back(num / i);
		}
	}
	sort(vec.begin(), vec.end());
}
