#include <iostream>

struct student{
	long num;
	float score;
	struct student *next;
};

using namespace std;
int main(int argc, char *argv[]) {
	struct student a,b,c,*head,*p;
	a.num=2313;a.score=12.23;
	b.num=213l;b.score=1312;
	c.num=123;c.score=231;
	head= &a;
	a.next=&b;
	b.next=&c;
	c.next=NULL;
	p=head;
	do {
		cout<<p->num<<" "<<p->score<<endl;
		p=p->next;
	} while (p!=NULL);
	return 0;	
}