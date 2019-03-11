#include <iostream>
#include <cstdlib>
#include <stack>
using namespace std;
int main()
{
    stack<int> S;
    int a, b, x;
    string s;
    while (cin >> s)
    {
        if (s[0] == '+')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a + b);
        }
        if (s[0] == '-')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a - b);
        }
        if (s[0] == '*')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a * b);
        }
        if (s[0] == '/')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a / b);
        }
        if (atoi(s.c_str()) > 0)
        {
            S.push(atoi(s.c_str()))
        }
    }
    cout << S.top() << endl;
    return 0;
}