#include<iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <algorithm>
using namespace std;
#define MAX 50000
#define SENTINEL 2000000000 
int L[MAX/2+2],R[MAX/2+2];
int cnt;
int nxd=0;
long long ans;
int  merge(int A[],int n,int left,int mid,int right,long long *count){
	int n1=mid-left;
	int n2=right-mid;
	for (int i=0;i<n1;i++) {L[i]=A[left+i];}
	for (int i=0;i<n2;i++) {R[i]=A[mid+i];}
	L[n1]=R[n2]=SENTINEL;
	int i=0,j=0;
	for (int k=left;k<right;k++) {
		cnt++;
		if(L[i]<=R[j]){
			A[k]=L[i++];
			(*count) = (*count) + mid - i + 1;
		}else {
			A[k]=R[j++];
		}
	}
	return left+right-nxd;
}

void mergeSort(int A[],int n,int left,int right,long long *count){
	if(left+1<right){
		int mid=left+right;
		mid/=2;
		mergeSort(A, n, left, mid,count);
		mergeSort(A, n, mid, right,count);
		merge(A, n, left, mid, right,count);
	}
}

int main(){
	int A[MAX],n,i;
	cnt=0;
	cin>>n;
	for (i=0;i<n;i++) {
		cin>>A[i];
	}
	mergeSort(A, n, 0, n,&ans);
	for (i=0;i<n;i++) {
		cout<<A[i]<<" ";
	}
	cout<<endl<<"nxd="<<ans;
	return 0;
}