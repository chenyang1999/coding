#include <iostream>
#include <map>
#include <string>
using namespace std;
void print(map<string, int>T){
	map<string, int>::iterator it;
	cout<<T.size()<<endl;
	for (it =T.begin();it!=T.end();it++) {
		pair<string, int >item = *it;
		cout<<item.first<<"-->"<<item.second<<endl;	
	}
}

int main(int argc, char *argv[]) {
	map<string, int>T;
	T["red"]=32;
	T["blue"]=688;
	T["yellow"]=122;
	T["blue"]+=312;
	print(T);
	T.insert(make_pair("zebra", 101010));
	T.insert(make_pair("white", 0));
	T.erase("yellow");
	print(T);
	pair<string, int >target = *T.find("red");
	cout<<target.first<<"-->"<<target.second<<endl;
	return 0;	
}