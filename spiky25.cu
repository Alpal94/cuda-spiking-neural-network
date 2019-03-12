#define no_input_neurons 12
#define no_output_neurons 3
#define bSize 30

/*
 * OUTPUT Neurons:
 * 0,1,2 ... 7,8 --> retina vision sensors
 * 9 --> food touch sensor
 * 10 --> pain touch sensor
 * 11 --> signal generator
 *
 * INPUT Neurons:
 * 12 --> forward motor neuron (output)
 * 13 --> right motor neuron (output)
 * 14 --> left motor neuron (output)
 *
 */

class SPIKY25 {
	private:
	int board[bSize][bSize];
	int **retina;
	int posX;
	int posY;

	void insertFood(int x, int y) {
		if(y+1 >= bSize || x+1 >= bSize)  return;
		if(board[x][y] || board[x][y+1] || board[x+1][y] || board[x+1][y+1]) return;
		for(int i = x -1; i < 3; i++) {
			for(int j = y - 1; j < 3; j++) {
				if(i > 0 && i < bSize && j > 0 && j < bSize)
					if(board[i][j]) return;
			}
		}

		board[x][y] = 1;
		board[x][y+1] = 0;
		board[x+1][y]  = 1;
		board[x+1][y+1] = 1;
	}
	void insertSpiky(int x, int y) {

		if(y+1 >= bSize || x+1 >= bSize)  return;
		if(board[x][y] || board[x][y+1] || board[x+1][y] || board[x+1][y+1]) return;
		for(int i = x -1; i < 3; i++) {
			for(int j = y - 1; j < 3; j++) {
				if(i > 0 && i < bSize && j > 0 && j < bSize)
					if(board[i][j]) return;
			}
		}

		board[x][y] = 1;
		board[x][y+1] = 1;
		board[x+1][y]  = 1;
		board[x+1][y+1] = 0;
	}

	public:
	void init() {
		retina = (int**) malloc (3*sizeof(int));
		for(int i = 0; i < 3; i++) {
			retina[i] = (int*)malloc(3*sizeof(int));
		}
		posX = 15;
		posY = 0;
		for(int i = 0; i < bSize; i++) 
			for(int j = 0; j < bSize; j++) 
				board[i][j] = 0;
		/*insertFood(4, 8);
		insertFood(5, 13);
		insertFood(20, 10);
		insertFood(10, 5);
		insertFood(0, 2);*/
		insertFood(posX - 5, posY + 1);
		insertFood(posX + 5, posY + 1);
		insertFood(posX, posY);
		insertFood(posX - 10, posY + 1);
		insertFood(posX + 10, posY + 1);

		/*insertSpiky(3, 0);
		insertSpiky(25, 20);
		insertSpiky(16, 10);
		insertSpiky(8, 3);
		insertSpiky(0, 8);*/
		
		for(int i = 0; i < 30; i++) {
			insertFood(rand() % 30, rand() % 30);
			insertSpiky(rand() % 30, rand() % 30);
		}
		printBoard();
			
	}

	void move(bool forward, bool right, bool left) {
		if(forward) {
			if(posY + 1 < bSize) posY++;
			else posY = 0;
		} else if(right) {
			if(posX + 1 < bSize - 1) posX++;
			else posX = 0;
		} else {
			if(posX - 1 > 1) posX--;
			else posX = bSize - 1;
		}
	}
	int touch() {
		for(int i = posX - 1; i < posX + 1; i++) {
			for(int j = posY; j < posY + 2; j++) {
				if(board[i][j] && board[i][j+1] && board[i+1][j] && !board[i+1][j+1]) {
					return -1; // pain
				}
				if(board[i][j] && !board[i][j+1] && board[i+1][j] && board[i+1][j+1]) {
					return 1; // food
				}
			}
		}
		return 0;
	}
	int **vision() {
		for(int i = posX - 1; i < posX + 2; i++) {
			for(int j = posY; j < posY + 3; j++) {
				retina[i - posX + 1][j - posY] = board[i][j];
			}
		}
		return retina;
	}

	void printBoard() {
		for(int i = 0; i < bSize; i++) {
			for(int j = 0; j < bSize; j++) {
				if(posX == i && posY == j) printf("*");
				else printf("%d", board[i][j]);
			}
			printf("_\n");
		}
		printf("\n_\n");
	}
};
