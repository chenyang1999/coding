#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
#define MAX 200000
#define INFTY (1<<30)
int H,A[MAX+1];
void maxHeapify(int i){
//	9_B
}
int extract(){
	int maxv;
	if(H<1)return -INFTY;
	maxv=A[1];
	A[1]=A[H--];
	maxHeapify(1);
	return maxv;
}
void increaseKey(int i,int key){
	if(key< A[i])return;
	A[i]=key;
	while (i>1 && A[i/2]<A[i]) {
		swap(A[i], A[i/2]);
		i=i/2;
	}
}
void insert(int key){
	H++;
	A[H]=-INFTY;
	increaseKey(H, key);
}

int main(int argc, char *argv[]) {
	int key;
	char com[10];
	while (1) {
		cin>>com;
		if(com[0]=='e'  && com[1] =='n')break;
		if(com[0]=='i'){
			cin>>key;
			insert(key);
		}else {
			cout<<extract();
		}
	}
	return 0;	
}