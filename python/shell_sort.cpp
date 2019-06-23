#include <iostream> 
using namespace std;
/* function to sort arr using shellSort */
#define N 10000000
int arr[N];
#include <time.h>
void shellSort(int arr[], int n) 
{ 
	// Start with a big gap, then reduce the gap 
	for (int gap = n / 2; gap > 0; gap /= 2) { 
		// Do a gapped insertion sort for this gap size. 
		// The first gap elements arr[0..gap-1] are already in gapped order 
		// keep adding one more element until the entire array is 
		// gap sorted 
		for (int i = gap; i < n; i += 1) { 
			// add arr[i] to the elements that have been gap sorted 
			// save arr[i] in temp and make a hole at position i 
			int temp = arr[i]; 
  
			// shift earlier gap-sorted elements up until the correct 
			// location for arr[i] is found 
			int j; 
			for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) 
				arr[j] = arr[j - gap]; 
  
			// put temp (the original arr[i]) in its correct location 
			arr[j] = temp; 
		} 
	} 
} 
  
void printArray(int arr[], int n) 
{ 
	for (int i = 0; i < n; i++) 
		cout << arr[i] << " "; 
	cout << "\n"; 
} 
  
int main() 
{ 
	for (int i=0;i<N;i++) {
		arr[i]=rand()%750;	
		} 
	int n = sizeof(arr) / sizeof(arr[0]); 
	cout << "Array before sorting: \n"; 
//	printArray(arr, n); 
	clock_t t1 = clock();
	shellSort(arr, n); 
	cout << "Array after sorting: \n"; 
	cout << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
//	printArray(arr, n); 
} 