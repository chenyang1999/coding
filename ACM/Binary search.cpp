#include <iostream>

using namespace std;
int A[100000],n;
//Binary search
int binarySearch(int key){
	int left,right,mid;
	while (left<right) {
		mid=(left+right)/2;
		if(key==A[mid])return 1;
		if(key>A[mid])left=mid+1;
		else if(key<A[mid])right=mid;
	}
	return 0;
}
int main(int argc, char *argv[]) {
	int i,q,k,sum;
	cin>>n;
	for (int i=0;i<n;i++) {
		cin>>A[i];
	}
	cin>>q;
	while (q--) {
		cin>>k;
		if(binarySearch(k))sum++;
	}
	cout<<sum;
	return 0;
	
	
}