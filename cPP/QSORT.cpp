#include <iostream>
#include <stack>
using namespace std;
#include <time.h>

#include <algorithm>
#define N 100000000
int arr[N];
void swapElement(int& a,int& b)
{
	
	int temp=a;
	a=b;
	b=temp;
 
}
void printArray(int arr[], int n) 
{ 
	for (int i = 0; i < n; i++) 
		cout << arr[i] << " "; 
	cout << "\n"; 
} 
//非递归的形式
void QuickSortNonRecursive(int arr[],int length)
{
	stack<int> lowHigh;//先存大再存小，取得时候就可以先取小再取大，此处的大小指的是数组索引
	lowHigh.push(length-1);
	lowHigh.push(0);
	int low, high;
	while(!lowHigh.empty())
	{
		low=lowHigh.top();lowHigh.pop();
		high=lowHigh.top();lowHigh.pop();
		if(low>=high)continue;
		int i=low;int j=high;
		int value=arr[low];
		while(i<j)//i==j循环结束
		{
			
			while(arr[j]>value)j--;//右边的都大于value
			swapElement(arr[j],arr[i]);
			while(arr[i]<value)i++;//左边的都小于value
			swapElement(arr[i],arr[j]);
		}
		//开始存储左右两侧待处理的数据，为了先处理左侧先保存右侧数据
		lowHigh.push(high);
		lowHigh.push(i+1);
		//左侧
		lowHigh.push(i-1);
		lowHigh.push(low);
		
	}
	
}
int main(int argc, char *argv[]) {

	
//	srand((unsigned)time(NULL));

	for (int i=0;i<N;i++) {
		arr[i]=rand()%750;
	} 
	int n = sizeof(arr) / sizeof(arr[0]); 
	cout << "Array before sorting: \n"; 
//	printArray(arr, n); 
	clock_t t1 = clock();

	sort(arr,arr+n); 
	cout << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
	cout << "Array after sorting: \n"; 
//	printArray(arr, n); 

}