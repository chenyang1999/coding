//
//  main.cpp
//  numberSequence_hdu
//
//  Created by Alps on 14/12/22.
//  Copyright (c) 2014年 chen. All rights reserved.
//
 
#include <iostream>
using namespace std;
 
 
 
void multiMatrix(int ma[][2],int a, int b){
	int i,j;
	int cp[2][2] = {0,0,0,0};;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			cp[i][j] = ((ma[i][0]*ma[0][j])%7 + (ma[i][1]*ma[1][j])%7)%7;
//            printf("%d ",cp[i][j]);
		}
//        printf("\n");
	}
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			ma[i][j] = cp[i][j];
		}
	}
}
 
void multiDoubleMatrix(int cp[][2], int ma[][2], int a, int b){
	int temp[2][2];
	int i,j;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			temp[i][j] = ((cp[i][0]*ma[0][j])%7 + (cp[i][1]*ma[1][j])%7)%7;
		}
	}
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			cp[i][j] = temp[i][j];
		}
	}
}
 
int calculate(int ma[][2], int a, int b, int c){
	if (c <= 0) {
		return 1;
	}
	int cp[2][2] = {1,0,0,1};
	while (c) {
		if (c&1) {
			multiDoubleMatrix(cp, ma, a, b);
		}
		multiMatrix(ma, a, b);
		c = c>>1;
	}
	return (cp[0][0]+cp[0][1])%7;
}
 
int main(int argc, const char * argv[]) {
	int a,b,c;
	while (1) {
		scanf("%d %d %d",&a,&b,&c);
		int ma[][2] = {a%7,b%7,1,0};
//        printf("%d %d %d %d\n",ma[0][0],ma[0][1],ma[1][0],ma[1][1]);
		if (a == 0 && b == 0 && c == 0) {
			break;
		}
		printf("%d\n",calculate(ma, a, b, c-2));
	}
	
	return 0;
}
 
