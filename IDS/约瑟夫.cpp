#include <iostream>
// Joseph ring

struct Node{
	int data;
	struct Node *next;
	
};


using namespace std;
int main(int argc, char *argv[]) {
	struct Node *head ,*s,*q,*t;
	
	int n,m,count=0;
	cin>>n>>m;
	for (int i=1;i<=n;i++) {
		s=(struct Node *)malloc(sizeof(struct Node));
		s->data=i;
		s->next=NULL;
		
		if(i==1){
			head=s;
			q=head;
			
		}
		else {
			q->next=s;
			q=q->next;
			
		}
	}	
	
	q->next=head;
	cout<<"before out of queue";
	q=head;
	
	while (q->next!=head) {
		cout<<q->data;
		q=q->next;
		
	}
	cout<<q->data<<endl;
	
	cout<<"after out of queue";
	
	q=head;
	do {
		count++;
		if(count==m-1){
			t=q->next;
			q->next=t->next;
			count=0;
			cout<<t->data<<" ";
		}
		q=q->next;
	} while (q->next!=q);
	
	cout<<q->data;
	return 0;
}