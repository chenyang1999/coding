#include <iostream>
#include <list>
#include <algorithm>
using namespace std;

int main(int argc, char *argv[]) {
	int q,x;
	char com[20];
	list<int>v;
	cin>>q;
	for (int i=0;;i<q;i++) {
		cin>>com;
		if(com[0]=='i'){
			cin>>x;
			v.push_front(x);
		}else if (com[6]=='L'){
			v.pop_back();
		}else if(com[6]=='F'){
			v.pop_front();
		}else if (com[0]=='d'){
			cin>>X;
			for (list<int>::iterator it =v.begin();if!=v.end();it++) {
				if(*it ==x){
					v.erase(it);
					break;
				}
			}
		}
	}	
	int i=0;
for (list<int>::iterator it=v.begin();if!=v.end();it++) {
		if(i++)printf("  ");
		printf("%d",*it);
		
	}
	cout<<endl;
	return 0;
}