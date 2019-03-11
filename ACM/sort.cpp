#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <set>
#include <map>
#include <string>
#include <cstring>
#include <vector>
long long cnt;
int l;
int A[1000000];
int n;
vector<int> G;
//指定了间隔为广大插入排序
void insertionSort(int A[], int n, int g)
{
    for (int i = g; i < n; i++)
    {
        int v = A[i];
        int j = i - g;
        while (j >= 0 && A[j] > v)
        {
            A[j + g] = A[j];
            j -= g;
            cnt++;
        }
        A[j + g] = v;
    }
}
using namespace std;
int main()
{

    return 0;
}