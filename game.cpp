#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>

#define no_input_neurons 5
#define no_output_neurons 10

bool moveEvil();
bool moveGood( bool l, bool s, bool r );
void printBoard();


char *input;
char *output;
int xE = 3; int yE = 4;
int xC = 4; int yC = 0;
int board_size = 5;

int main() {
	while(true) {
		printBoard();
		char c = getchar();
		bool res;
		if(c == 'a') res = moveGood(true, false, false);
		else if(c == 's') res = moveGood(false, true, false);
		else if(c == 'd') res = moveGood(false, false, true);
		if(res) { printf("Good wins!  Game over\n"); break; }
			
		res = moveEvil();
		if(res) { printf("Evil wins!  Game over\n"); break; }
	}

}


void printBoard() {
	for(int i = 0; i < board_size; i++) {
		for(int j = 0; j < board_size; j++) {
			if(i == xC && j == yC) printf("%c", 'C');
			else if(i == xE && j == yE) printf("%c", 'E');
			else printf(" ");
		}
		printf("\n");
	}
}

bool moveEvil() {
	yE--;
	if ( yE < 0 ) return true;
	else return false;
}
bool moveGood( bool l, bool s, bool r ) {
	if(l && xC - 1 >=0) xC--;
	if(r && xC + 1 < 5) xC++;
	if(xC == xE && yC == yE) return true;
	else return false;
}
