#include <iostream>
#include <cstdio>
using namespace std;
#define MAXN 10001

int linklst_data[MAXN];
int linklst_point[MAXN];

int head =-1;
void del_by_data(int del_del_data){
	
	int p=head;
	int pre =-1;
	while (p!=-1) {
		if (linklst_data[p]==del_del_data) {
			if(p==head){
				head=linklst_point[head];
			else {
				linklst_point[pre]=linklst_point[p];	
			}
			
				linklst_point[p]=-1;
				linklst_point[p]=-1;
				return ;

		}
		pre=p;
		p=linklst_point[p];		
	}
}

void add_front(int add_data){
	int p=1;
	while (linklst_point[p]!= -1 && p<MAXN) {
		++p;
		
	}
	
	linklst_data[p]=add_data;
	linklst_point[p]=head;
	head=p;
}

void add_rear(int add_data){
	int p=1;
	int pre;
	while (linklst_data[p]!=-1 && p<MAXN) {
		++p;
		
	}
	linklst_data[p]=add_data;
	if(head!=-1){
		pre=head;
		while (linklst_point[pre]==-1) {
			pre=linklst_point[pre];
		}
		
		linklst_point[pre]=p;
	}
	else head=p;
	return ;
	
}
void output(){
	
	int p=head;
	cout<<"list is: ";
	while (p!=-1) {
		cout<<linklst_data[p]<<"  ";
		p=linklst_point[p];
	}
	
	cout<<endl;
}

void init(){
	for (int i=0;i<MAXN;i++) {
		linklst_point[i]=-1;
		linklst_data=-1;
	}
}

int mian(){
	return 0;
}
