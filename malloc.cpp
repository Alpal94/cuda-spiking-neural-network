#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main() {
	int **pointer = (int**) malloc(2*sizeof(int*));
	for(int i = 0; i < 2; i++) pointer[i] = (int*) malloc(2*sizeof(int));
	for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) pointer[i][j] = 4;

	for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) printf(" %d ", pointer[i][j]);
}
