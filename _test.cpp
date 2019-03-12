#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main() {
	//Read in neural network
	char buff[255];
	FILE *fp = fopen("network", "r");
	char *line = NULL; size_t len = 0; ssize_t read;

	int noLines = 0;
	while( (read = getline(&line, &len, fp)) != -1) noLines++;
	fclose(fp);
	fp = fopen("network", "r");

	int **neighbors = (int **) malloc( sizeof(int) * noLines );
	noLines = 0;

	int i = 0; int j = 0;
	while( (read = getline(&line, &len, fp)) != -1) {
		char *cpyline = "";
		strcpy(cpyline, line);
		char *nodes = strtok(line, " ");

		int noNodes = 0;
		int nodeList[BUFSIZ];

		while(nodes) {
			int node = atoi(nodes);
			//puts(nodes);
			nodes = strtok(NULL, " ");
		}
		i++; j=0;

		nodes = strtok(cpyline, " ");
		neighbors[i] = (int*) malloc(sizeof(int) * noNodes);
		while(nodes) {
			int node = atoi(nodes);
			neighbors[i][j] = node;

			//printf("I: %d J: %d Node: %d\n", i, j, node);
			puts(nodes);
			nodes = strtok(NULL, " ");
			j++;
		}
	}
}
