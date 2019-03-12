#include <iostream>
#include <set>
using namespace std;
void print(set<int>S){
	cout<<S.size()<<" : "<<endl;
	for (set<int >::iterator it=S.begin();it!=S.end();it++) {
		cout<<" "<<(*it);
	}
	cout<<endl;
}
int main(int argc, char *argv[]) {
	set<int >S;
	S.insert(9);
	S.insert(1);
	S.insert(7);
	S.insert(8);
	print(S);
	S.erase(7);
	S.insert(2);
	S.insert(10);
	print(S);
	return 0;
}