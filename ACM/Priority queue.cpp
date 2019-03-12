#include <iostream>
#include <queue>
using namespace std;
int main(int argc, char *argv[]) {
	priority_queue<int >PQ;
	PQ.push(1);
	PQ.push(2);
	PQ.push(8);
	PQ.push(5);
	cout<<PQ.top()<<" ";
	PQ.pop();
	return 0;
}