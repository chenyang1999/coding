#include <iostream>
#include <cstdlib>
#include <cstdio>

#define  ElemType int 
typedef  struct List *link;
typedef  struct List Lnode;
using namespace  std;

struct List{
	ElemType data;
	struct List *next;
	
};

link setnull(link Head){
	link p;
	p=Head;
	while (p!=NULL) {
		p=p->next;
		free(Head);
		Head=p;
	}
	return Head;
	
}

link insert(link Head,ElemType x,int i){
	link NewPoint,p=Head;
	int j=1;
	NewPoint =(link)malloc(sizeof(Lnode));
	NewPoint->data=x;
	if(i==1)	{
		NewPoint->next=Head;
		Head=NewPoint;
		
	}
	else 
	{
		while (j<i-1 && p->next!=NULL) {
			p=p->next;
			j++;
		}
		if (j==i-1) {
			NewPoint->next =p->next;
			p->next=NewPoint;
			
		}
		else {
			printf("insert is Failure,i is not right");
			
		}
	}
	
	return Head;
	
}

link creat(link Head){
	ElemType newData;
	link NewPoint;
	
	Head=(link)malloc(sizeof(Lnode));
	printf("please input number :\n");
	scanf("%d",&newData);
	Head->data =newData;
	Head->next =NULL;
	while (1) {
		NewPoint=(link)malloc(sizeof(Lnode));
		if(NewPoint ==NULL){
			break;
		}
		else{
			printf("input a number ,-1 means exit it \n");
			scanf("%d",&newData);
			if(newData==-1)
				return Head;
			else {
				NewPoint->data=NewPoint;
				NewPoint->next=Head;
				head=NewPoint;
				
			}
		}
	}
	return Head;
}

int lenth(link Head){
	int len =0;
	link p;
	p=Head;
	while (p!=NULL){
		len++;
		p=p->next;
	}
	return len;
	
}

ElemType get(link Head ,int i){
	int j=1;
	link p;
	p=Head;
	while (j<i && p!=NULL) {
		p=p->next;
		j++;
		
	}	
	if (p!=NULL) {
		return (p->data);
		
	}
	else {
		printf("data is error\n");
		return -1;
	}
}

int locate(link Head,ElemType x){
	int n=0;
	link p;
	p=Head;
	while (p!=NULL && p->data!=x) {
		p=p->next;
		n++;
		
	}
	if (p==NULL) {
		return -1;
		
	}
	else {
		return n+1;
	}
}

void display(link Head){
	
	link p;
	p=Head;
	if(p==NULL){
		printf("\nList is empty\n");
	}
	else {
		do {
			printf("%d ",p->data);
			p=p->next;
		} while (p!=NULL)
	}
}

link connect(link Head1,link Head2){
//connect H2 to H1's tail
	link p;
	p=Head1;
	
	while (p->next!=NULL) {
		p=p=>next;
	}
	p->next=Head2;
	return Head1;
}

link del(link Head ,int i){
	int j=1;
	link p;
	link t;
	p=Head;
	if(i==1){
		p=p->next;
		free(Head);
		Head=p;
		
	}
	else {
		while (j<-i-1 && p->next!=NULL) {
			p=p->next;
			j++;
			
		}
		if (p->next !=NULL && j== i-1) {
			t=p->next;
			p->next=t->next;
			
		}
		if(t!=NULL){
			free(t);
		}
	}
	
	return Head;
}

int compare(link Head1,link Head2){
	
	link p1,p2;
	p1=Head1;
	p2=Head2;
	while (1) {
		if(p1->next ==NULL && p2->next== NULL){
			return 1;
			
		}
		if (p1->data!=p2->data) {
			return 0;
		}
		else {
			p1=p1->next;
			p2=p2->next;
		}
	}
}

int main(int argc, char *argv[]) {
	return 0;	
}