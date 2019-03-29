#include <iostream>
#define MAXN 10001

using namespace std;

int link_data[MAXN];
int link_point[MAXN];
int stack[MAXN];
int head =-1;
int stack_head=0;

void del_by_data(int del_data){
	int p=head;
	int pre=-1;
	while (p!=-1) {
		if(link_data[p] ==del_data){
			
			if(p==head){
				head=link_point[head];	
			}
			else {
				link_point[pre]=link_point[p];
			}
			
			link_data[p]=-1;
			link_point[p]=-1;
			stack[--stack_head]=p;
			return ;
			
		}
		pre=p;
		p=link_point[p];
		
	}
	return ;
}

void add_front(int add_data){
	int p=1;
	p=stack[stack_head++];
	link_data[p]=add_data;
	link_point[p]=head;	
	head=p;
	
}

void add_rear(int add_data){
	int p=1;
	int pre;
	p=stack[stack_head++];
	link_data[p]=add_data;
	if(head!=-1){
		pre=head;
		while (link_point[pre]!=-1) {
			pre=link_point[pre];
			link_point[pre]=p;
			
		}
	}
	else {
		head=p;
		
	}
}


void output(){
	int p=head;
	while (p!=-1) {
		cout<<link_data[p]<<" ";
		p=link_point[p];
	}
	
	cout<<endl;
	return ;
}

void init(){
	for (int i=0;i<MAXN;i++) {
		link_point[i]=-1;
		link_data[i]=-1;
		stack[i]=i+1;
		
	}
	return ;
}
int main(int argc, char *argv[]) {
		
}