#include <iostream>
#include <stack>
#include <time.h>

using namespace std;
#define N 12
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
//递归的形式
void QuickSortRecursive(int arr[],int low,int high)
{
 
	if(low>=high)return;
	int value=arr[low];
	int i=low;int j=high;
	while(i<j)//i==j循环结束
	{
		
		while(arr[j]>value)
			j--;//右边的都大于value
		swapElement(arr[j],arr[i]);
		while(arr[i]<value)
			i++;//左边的都小于value
		swapElement(arr[i],arr[j]);
	}
	QuickSortRecursive(arr,low,i-1);//左侧快排
	QuickSortRecursive(arr,i+1,high);//右侧快排
	
}
int main(int argc, char *argv[]) {

	
//	srand((unsigned)time(NULL));

	for (int i=0;i<N;i++) {
		arr[i]=rand()%1000;
	} 
	int n = sizeof(arr) / sizeof(arr[0]); 
	cout<<n;
	cout << "Array before sorting: \n"; 
	printArray(arr, n); 
	QuickSortRecursive(arr,0,n-1); 
	cout << "Array after sorting: \n"; 
	printArray(arr, n); 

}