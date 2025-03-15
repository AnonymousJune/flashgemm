#include <math.h>
#include <algorithm>
#include <limits.h>
using namespace std;

long mb, me, nb, ne, costM, costN;

int costM12 = 12;
int COST_M8 = 9;
int COST_M4 = 6;
int COST_M1 = 6;

int COST_N32 = 32;
int COST_N16 = 20;
int COST_N8 = 18;
int COST_N4 = 17;
int costN1 = 16;

void MblockCost(int mb1, long M, int Tm)
{
	int me1 = M - mb1 * (Tm - 1);
	if (me1 <= 0)
	{
		return;
	}
	if (me1 == mb1)
		me1 = 0;
	long cost_mb, cost_me, costMmax;
	cost_mb = mb1 / 12 * costM12 + (mb1 % 12) / 8 * COST_M8 + ((mb1 % 12) % 8) / 4 * COST_M4 + (mb1 % 12) % 4 * COST_M1;
	cost_me = me1 / 12 * costM12 + (me1 % 12) / 8 * COST_M8 + ((me1 % 12) % 8) / 4 * COST_M4 + (me1 % 12) % 4 * COST_M1;
	costMmax = max(cost_mb, cost_me);
	if (costMmax < costM)
	{
		costM = costMmax;
		mb = mb1;
		me = me1;
	}
}

void setMBlock(long M, int Tm)
{
	mb = (M + Tm - 1) / Tm;
	me = M % mb;
	if (Tm == 1)
		return;
	costM = LONG_MAX;
	MblockCost(mb, M, Tm);
	switch (mb % 12)
	{
	case 1:
		MblockCost(mb + 3, M, Tm);
		MblockCost(mb + 7, M, Tm);
		MblockCost(mb + 11, M, Tm);
		break;
	case 2:
		MblockCost(mb + 2, M, Tm);
		MblockCost(mb + 6, M, Tm);
		MblockCost(mb + 10, M, Tm);
		break;
	case 3:
		MblockCost(mb + 1, M, Tm);
		MblockCost(mb + 5, M, Tm);
		MblockCost(mb + 9, M, Tm);
		break;
	case 4:
		MblockCost(mb + 4, M, Tm);
		MblockCost(mb + 8, M, Tm);
		break;
	case 5:
		MblockCost(mb + 3, M, Tm);
		MblockCost(mb + 7, M, Tm);
		break;
	case 6:
		MblockCost(mb + 2, M, Tm);
		MblockCost(mb + 6, M, Tm);
		break;
	case 7:
		MblockCost(mb + 1, M, Tm);
		MblockCost(mb + 5, M, Tm);
		break;
	case 8:
		MblockCost(mb + 4, M, Tm);
		break;
	case 9:
		MblockCost(mb + 3, M, Tm);
		break;
	case 10:
		MblockCost(mb + 2, M, Tm);
		break;
	case 11:
		MblockCost(mb + 1, M, Tm);
		break;
	default:
		break;
	}
}


void NblockCost(int nb1, long N, int Tn)
{
	int ne1 = N - nb1 * (Tn - 1);
	if (ne1 <= 0)
	{
		return;
	}
	if (ne1 == nb1)
		ne1 = 0;
	long cost_nb, cost_ne, costNmax;
	// printf("nb1=%-5d N1:%-5d\n",nb1,(nb1 % 4) * costN1);
	cost_nb = nb1 / 32 * COST_N32 + (nb1 % 32) / 16 * COST_N16 + (nb1 % 16) / 8 * COST_N8 + (nb1 % 8) / 4 * COST_N4 + (nb1 % 4) * costN1;
	cost_ne = ne1 / 32 * COST_N32 + (ne1 % 32) / 16 * COST_N16 + (ne1 % 16) / 8 * COST_N8 + (ne1 % 8) / 4 * COST_N4 + (ne1 % 4) * costN1;
	costNmax = max(cost_nb, cost_ne);
	if (costNmax < costN)
	{
		costN = costNmax;
		nb = nb1;
		ne = ne1;
	}
}

void setNBlock(long N, int Tn)
{
	nb = (N + Tn - 1) / Tn;
	ne = N % nb;
	if (Tn == 1)
		return;
	costN = LONG_MAX;
	NblockCost(nb, N, Tn);
	switch (nb % 16)
	{
	case 1:
		NblockCost(nb + 3, N, Tn);
		NblockCost(nb + 7, N, Tn);
		NblockCost(nb + 11, N, Tn);
		NblockCost(nb + 15, N, Tn);
		break;
	case 2:
		NblockCost(nb + 2, N, Tn);
		NblockCost(nb + 6, N, Tn);
		NblockCost(nb + 10, N, Tn);
		NblockCost(nb + 14, N, Tn);
		break;
	case 3:
		NblockCost(nb + 1, N, Tn);
		NblockCost(nb + 5, N, Tn);
		NblockCost(nb + 9, N, Tn);
		NblockCost(nb + 13, N, Tn);
		break;
	case 4:
		NblockCost(nb + 4, N, Tn);
		NblockCost(nb + 8, N, Tn);
		NblockCost(nb + 12, N, Tn);
		break;
	case 5:
		NblockCost(nb + 3, N, Tn);
		NblockCost(nb + 7, N, Tn);
		NblockCost(nb + 11, N, Tn);
		break;
	case 6:
		NblockCost(nb + 2, N, Tn);
		NblockCost(nb + 6, N, Tn);
		NblockCost(nb + 10, N, Tn);
		break;
	case 7:
		NblockCost(nb + 1, N, Tn);
		NblockCost(nb + 5, N, Tn);
		NblockCost(nb + 9, N, Tn);
		break;
	case 8:
		NblockCost(nb + 8, N, Tn);
		NblockCost(nb + 4, N, Tn);
		break;
	case 9:
		NblockCost(nb + 7, N, Tn);
		NblockCost(nb + 3, N, Tn);
		break;
	case 10:
		NblockCost(nb + 6, N, Tn);
		NblockCost(nb + 2, N, Tn);
		break;
	case 11:
		NblockCost(nb + 5, N, Tn);
		NblockCost(nb + 1, N, Tn);
		break;
	case 12:
		NblockCost(nb + 4, N, Tn);
		break;
	case 13:
		NblockCost(nb + 3, N, Tn);
		break;
	case 14:
		NblockCost(nb + 2, N, Tn);
		break;
	case 15:
		NblockCost(nb + 1, N, Tn);
		break;
	default:
		break;
	}
}