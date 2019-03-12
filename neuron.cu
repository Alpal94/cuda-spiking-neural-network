#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "spiky25.cu"

#define neurons_per_thread 50
#define no_input_neurons 12
#define no_output_neurons 3
#define clock_cycle 10

int **getNeighbors(char *file);

//This is an optimised queue for this project
struct Queue {
	private:
	unsigned short int *_queue;
	short int front; //Better for front and back to be signed, as modular arithmetic is more efficient
	short int back;
	unsigned short int size;

	public:
	unsigned short int neuron_count;

	public:
	__device__ void init(unsigned short int *sPtr, int queue_size, unsigned short int neuron) {
		_queue = sPtr;
		front = 0;
		back = 0;
		size = queue_size;
		_queue[front] = neuron;
	}

	__device__ short int push(unsigned short int neuron) {
		//printf("Pushing: %d\n", neuron);
		//printf(" ");
		back = (back - 1) % size;
		if(back < 0) back = size - 1;
		_queue[back] = neuron;
		return back;
	}

	__device__ unsigned short int sFront() {
		//printf("FRONT: %d - %d\n", front, _queue[front]);
		return _queue[front];
	}

	__device__ unsigned short int pop() {
		if(front > back) {
			short int tmp = front;
			front = (front - 1) % size;
			if(front < 0) front = size - 1;
			return _queue[tmp];
		} else if(front < back) {
			short int tmp = front;
			front = (front - 1) % size;
			if(front < 0) front = size - 1;
			return _queue[tmp];
		} else {
			unsigned short int tmp = _queue[front];
			_queue[front] = 0;
			return tmp;
		}
		
	}

	__device__ short int backPos() {
		return back;
	}

	__device__ short int frontPos() {
		return front;
	}

};

//This is the neuron
struct Neuron {
	private:
	unsigned short int *sharedPointer;
	int *oSodium, *iSodium;
	int noNodes;
	short int threshold;
	bool isSpiking = true;
	int **neighbors;
	struct Queue *queue;
	unsigned char *active_neurons;
	unsigned char *neuron_synapses;
	char *global_input;
	char *global_output;
	char *_clock;


	public:
	__device__ void init(unsigned short int *sPtr, int **_neighbors, struct Queue *_queue, unsigned char *_active_neurons, unsigned char *_neuron_synapses, char *input, char *output, char* clock) {
		//initialise variables
		neighbors = _neighbors;
		noNodes = neighbors[0][0];
		threshold  = 30;
		queue = _queue;
		active_neurons = _active_neurons;
		global_input = input;
		global_output = output;
		_clock = clock;
		neuron_synapses = _neuron_synapses;
		sharedPointer = sPtr;//Shared memory must be declared outside of the class, therefore we pass a pointer to it.
		if(blockDim.x * blockIdx.x + threadIdx.x > noNodes) { //If id of thread
			isSpiking = false;
			return;
		}

		//Initialise sodium concentrations
		//oSodium = &sharedPointer[2 * threadIdx.x + 0];
		//iSodium = &sharedPointer[2 * threadIdx.x + 1];
		//*oSodium = 0; *iSodium = 0;

		//Initialise neighbors

	}

	__device__ void run() {

		int x;
		x = threadIdx.x;
		printf("_%d_", x);
		if(blockIdx.x > 1) return;
		//if(blockDim.x * blockIdx.x + threadIdx.x > noNodes) return;
		short int sodium_c[neurons_per_thread];
		short int rec[neurons_per_thread]; //Keeps record of synapse weight index.
		for(short int i = 0; i < neurons_per_thread; i++) sodium_c[i] = 0;
		unsigned int counter = 0;

		while(true) {
			clock_t start = clock();
			clock_t now;
			for (;;) {
				if(_clock[0] < clock_cycle) {
					printf("");
					break;
				}
				/*now = clock();
				clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
				if (cycles >= 10000000) {
					break;
				}*/
			}
			if(threadIdx.x == 0 && blockIdx.x == 0) {
				printf("SPIKE");
				_clock[0]++;
			}
			__syncthreads();
			// Stored "now" in global memory here to prevent the compiler from optimizing away the entire loop.
			//printf("Start %d %d\n", threadIdx.x, now);
			//Grab global input  
			if(threadIdx.x < no_input_neurons && ( global_input[threadIdx.x] == 1 )) {
				queue->push((unsigned int) threadIdx.x + 1);
				//printf("Global input recieved: %d \n", threadIdx.x + 1);
			}
			//printf("end %d\n", threadIdx.x);

			if(threadIdx.x == 0) printf("Queue size: %d", -queue->backPos() + queue->frontPos());
			//First, determine which neurons are effected by the firing neuron, store this information in shared memory.  Node will be removed from the queue later.
			unsigned short int node = queue->sFront();
			if(node == 0) {
				//printf("No neurons\n");
				continue; //No neurons are spiking, wait for a spiking neuron. (Neuron == 0)
			}
			int *destinations = neighbors[(int) node]; //These are the neurons the sodium is sent to
			int noNeighbors = destinations[0]; //destinations[0] ==> refers to number of neighbors
			//printf("Node: %d Dest: %d %d %d\n", node, destinations[1], destinations[2], destinations[3]);

			int synapses_per_thread = neurons_per_thread; //12000 synapses max, all neurons must be processed in the same block, therefore  max 2048 threads available.
			for(int i = 0; i < synapses_per_thread; i++) {
				if(!(i + synapses_per_thread * threadIdx.x >  noNeighbors)) { 
					if(!(i + synapses_per_thread * threadIdx.x)) continue;
					int curr_neuron = destinations[synapses_per_thread * threadIdx.x + i];
					active_neurons[(curr_neuron - curr_neuron % 8)/8] += 1 << (curr_neuron % 8);
					//Store the synapse weights in shared memory
					//if(synapses_per_thread * threadIdx.x + i < sizeof(neuron_synapses))  --> could check overflow.  But if this happens, very bad so hopefully neural network won't be too big!
					neuron_synapses[synapses_per_thread * threadIdx.x + i] = (unsigned char) (destinations[(int) noNeighbors + synapses_per_thread * threadIdx.x + i]);
					printf("\nSynapse_ %d: %d\n", synapses_per_thread * threadIdx.x + i, neuron_synapses[synapses_per_thread * threadIdx.x + i]);

					//neurons_per_thread * 64 ==> Neurons per core
					//Record which neurons are to recieve sodium
					//This is a slow point which I tried very hard to avoid, uses up way to much memory and destroys efforts of the other parts of the program
					rec[ curr_neuron ] = synapses_per_thread * threadIdx.x + i;
				} else break; //syncthreads renders this unnecessary
			}

			//__syncthreads();   //Could be unecessary
			for(int i = 0; i < sizeof(neuron_synapses)/sizeof(*neuron_synapses); i++) { printf(" +=-%d", neuron_synapses[i]); }
			printf("+=-\n");

			if(threadIdx.x == 0 && blockIdx.x == 0) {
				queue->pop(); //Remove the active node from the queue, so that we do not reprocess it in the future.
				counter++; //Increment the counter so that we can determine old neurons
				printf("Counter: %d\n", counter);

			}

			__syncthreads();   //Could be unecessary

			//Now each neuron process the sodium it recieves.  We process more neurons per thread than synapses to maximise memory utilisation.
			if(!(neurons_per_thread * threadIdx.x >  noNeighbors)) {
				for(short int i = 0; i < neurons_per_thread; i++) {
					short int curr_neuron = neurons_per_thread * threadIdx.x + i;
					if(!curr_neuron) continue;
					//destinations[(int) node + neurons_per_thread * threadIdx.x + i];

					//printf("Curr: %d Active neurons: %d\n", curr_neuron, active_neurons[(int) (curr_neuron - curr_neuron % 8)/8] & 1 << (curr_neuron % 8));
					//printf("Hex: %x\n", active_neurons[(curr_neuron - curr_neuron % 8)/8]);
					if(active_neurons[(int) (curr_neuron - curr_neuron % 8)/8] & 1 << (curr_neuron % 8)) {
						//printf("ACTIVE: %d\n", curr_neuron);
						//This algorithm should be modified to improve machine learning, but this can be done after some experimental data.

						sodium_c[i] +=  synapseStrenghthCalc(counter, neuron_synapses[ rec[ curr_neuron] ]); 
						printf("ACTIVE: %d Strength: %d Sodium: %d Counter: %d\n", curr_neuron, neuron_synapses[ rec[ curr_neuron] ], sodium_c[i], counter);
						if(neuron_synapses[ rec[ curr_neuron] ] < 30) neuron_synapses[ rec[ curr_neuron] ]++;
						if(sodium_c[i] > threshold) {
							queue->push((unsigned short int) curr_neuron);
							printf("SPIKING: %d %d\n", curr_neuron, sodium_c[i]);
							if(curr_neuron >= no_input_neurons && curr_neuron < no_output_neurons + no_input_neurons) {
								printf("Setting global\n");
								global_output[curr_neuron]++;
								if(curr_neuron == 12) printf("MOVING FORWARD: %d\n", global_output[curr_neuron]);
							}
							/*printf("rec: ");
							for(int j = 0; j < sizeof(rec); j++) {
								printf("%d ", neuron_synapses[(int) rec[ j ]]);
							}
							printf("\n");*/
							sodium_c[i] = 0;
						}
						active_neurons[(curr_neuron - curr_neuron % 8)/8] -= 1 << (curr_neuron % 8); //Set it back to zero
					}
				}	
			}

			//Reset everything 
			for(int i = 0; i < synapses_per_thread; i++) {
				if(!(i + synapses_per_thread * threadIdx.x >  destinations[0])) {
					int curr_neuron = destinations[synapses_per_thread * threadIdx.x + i]; 
					active_neurons[(curr_neuron - curr_neuron % 8)/8] = 0b00000000;

					//Store the updated synapse weights in global memory
					destinations[(int) node + synapses_per_thread * threadIdx.x + i] = neuron_synapses[synapses_per_thread * threadIdx.x + i];
				}
			}
			//__syncthreads();   //Could be unecessary
		}
		//printf("\nFINISHED?\n");
	}

	__device__ void input(int sodium) {
		*oSodium += sodium;
	}

	__device__ void spiking() {
		if(*oSodium > threshold) {
			//send to neighbors
		}
	}

	__device__ unsigned char synapseStrenghthCalc(unsigned int counter, unsigned char synapseWeight) {
		return synapseWeight < 30 ? synapseWeight : 30;
		//printf("Counter: %d vs Weight: %d\n", counter, synapseWeight);
		if(counter < synapseWeight) {
			if(synapseWeight > counter) return (synapseWeight - counter < 10)?synapseWeight - counter:10;
			else return 1;
			
		} else {
			if(256 - counter + synapseWeight > 0) return (256 - counter + synapseWeight)?256 - counter + synapseWeight:10;
			else return 1;
		}
	}

};

__global__ void fcnCall(struct Neuron *nvidia, int **neighbors, struct Queue *queue, char *input, char *output, char* clock) {

	//Max memory is 48KB, a short int is 2Bytes.  Therefore, 24000 total of short ints available in shared memory.  Reserve every single byte.
   __shared__ unsigned short int nQueue[ 14904 ]; //Memory for queue of currently spiking neurons.  Make as big as possible. 
   __shared__ unsigned char active_neurons[ 8192 ]; //65536 bits, 65536 unique neurons per block.  Char has twice as many bits as a short int.
   __shared__ unsigned char neuron_synapses[ 10000 ]; //10000 synayses is the max number of synapses to be expected.  The strength of a synapse is represented by one byte, therefore reserve 10000 bytes for max 10000 synapses
			int x;
			x = threadIdx.x;
			printf("_%d_", x);

   for(int i = 0; i < sizeof(active_neurons); i++) {
	   active_neurons[i] = 0b00000000; //initialise active neuron array.
   }
   for(int i = 0; i < sizeof(neuron_synapses); i++) {
	   active_neurons[i] = 0b00000000; //initialise synapse array (probably not necessary)
   }

   if(threadIdx.x == 0 && blockIdx.x == 0) {
	   queue->init(nQueue, 500, 1);  
	   queue->push((unsigned short int) 5);
   }

		
   __syncthreads(); 

   nvidia->init(nQueue, neighbors, queue, active_neurons, neuron_synapses, input, output, clock);
   nvidia->run();


   return;
}

int main(void) {

	int **neighbors = getNeighbors("network");


	//Preparing these classes needs a wrapper
	struct Neuron *_neuron = (struct Neuron*) malloc(sizeof(*_neuron));
	struct Queue *_queue = (struct Queue*) malloc(sizeof(*_queue));

	struct Neuron *neuron;
	struct Queue *queue;

	char *input; char *output; char *clock;
	cudaMallocManaged(&input, no_input_neurons * sizeof(char));
	for (int i = 0; i < no_input_neurons; i++) input[i] = 0;
	cudaMallocManaged(&output, no_output_neurons * sizeof(int));
	for (int i = 0; i < no_output_neurons; i++) output[i] = 0;
	cudaMallocManaged(&clock, 2 * sizeof(char));

	cudaMalloc(&neuron, sizeof(*_neuron));
	cudaMemcpy(neuron, _neuron, sizeof(*_neuron), cudaMemcpyHostToDevice);
	cudaMalloc(&queue, sizeof(*_queue));
	cudaMemcpy(queue, _queue, sizeof(*_queue), cudaMemcpyHostToDevice);

	//Calls the function which calls the neuron
	fcnCall<<<1, 256>>>(neuron, neighbors, queue, input, output, clock);

	SPIKY25 spiky;
	spiky.init();
	int count = 0;
	clock[0] = 1;
	for(int i = 0;  ; i++) {
		if(clock[0] > clock_cycle - 1) {
			int touch = spiky.touch();
			int **vision = spiky.vision();
			for(int j = 0; j < no_input_neurons; j++) {
				input[j] = 0;
			}
			if(true) {
				if(touch > 0) {
					input[9] = 1;
					printf("Sending touch\n");
				} else if (touch < 0) {
					input[10] = 1;
					printf("Sending pain\n");
				}
			} else printf("NO PAIN"); 

			int j = 0;
			for(int a = 0; a < 3; a++) {
				for(int b = 0; b < 3; b++) {
					if(vision[a][b]) { input[j] = 1; printf("sending vision\n"); }
					j++;
				}
			}
			if(output[12] > 0) printf("REALLY ALERT MOVE: %d\n", output[12]);
			spiky.move(output[12]  > 0 ?  1 : 0, output[13] > 0 ? 2 : 0, output[14] > 0 ? 1 : 0);
			output[12] = 0;
			output[13] = 0;
			output[14] = 0;

			spiky.printBoard();
			printf("Count: %d\n", count);
			count++;
			if(count > 10000) exit(EXIT_SUCCESS);
			clock[0] = 0;
		}
	}

	//cudaMemcpy(_neuron, neuron, sizeof(*_neuron), cudaMemcpyDeviceToHost);
	//cudaMemcpy(_queue, queue, sizeof(*_queue), cudaMemcpyDeviceToHost);

	cudaFree(neuron);
	free(_neuron);
	cudaFree(queue);
	free(_queue);

	return 0;
}


/*
 * Returns pointer to adjacency list mapping out all of the neurons in the network.
 * Adjacency list is in the form of a jagged array, with each jagged array starting with a header int to describe array length.
 * First element [0][0] describes number of nodes in array.
 */

int **getNeighbors(char *file) {
	//Read in neural network
	char buff[255];
	FILE *fp = fopen(file, "r");
	char *line = NULL; size_t len = 0; ssize_t read;

	int noLines = 0;
	int **neighbors;
	noLines = 0;

	//first line is the header line, reads number of nodes.
	bool header = true;

	int i = 1; int j = 0;
	while( (read = getline(&line, &len, fp)) != -1) {
		char *nodes = strtok(line, " ");

		int noNodes = 0;
		int nodeList[BUFSIZ];

		bool first = true;
		j = 0; 
		int *temp; int *host_temp; //for cuda malloc copy
		while(nodes) {
			if(header) {
				int noLines = atoi(nodes);
				cudaMalloc( (void***) (&neighbors), sizeof(int*) * (noLines + 1) );
				cudaMalloc((void**) &(temp), sizeof(int) );
				//cudaMalloc((void**) (&neighbors[0][0]), sizeof(int) );
				cudaMemcpy(temp, &noLines, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(neighbors + 0, &temp, sizeof(int*), cudaMemcpyHostToDevice); 

				//cudaMallocManaged((void**) neighbors, sizeof(int*) * (noLines  + 1));
				//cudaMallocManaged((void*) (&neighbors[0][0]), sizeof(int) );
				//neighbors = (int**) malloc(noLines * sizeof(int*) + 1); //Add one for header int describing array length
				//neighbors[0] = (int*) malloc(sizeof(int) );
				//neighbors[0][0] = noLines; //record in header length of array

				header = false;
				nodes = strtok(NULL, " ");
				break;
			}
			else if(first) {
				noNodes = atoi(nodes);
				//free(host_temp);
				//cudaFree(temp);
				host_temp = (int*) malloc(sizeof(int) * 2 * noNodes + 1); //Add one for header int describing array length.  Multiply number of nodes by 2 as list includes synapse weights.
				cudaMalloc( (void**) &(temp), sizeof(int) * ( 2 * noNodes + 1) );
				
				host_temp[0] = noNodes;

				//cudaMallocManaged((void*) neighbors[i], sizeof(int) * (noNodes + 1)); //Add one for header int describing array length //neighbors[i] = (int*) malloc(sizeof(int) * noNodes + 1); //Add one for header int describing array length
				//neighbors[i][0] = noNodes; //record in header length of array

				j++;
				nodes = strtok(NULL, " ");
				first = false;
				continue;
			} else {
				int node = atoi(nodes);
				//cudaMemcpy(temp[j], node, sizeof(int), cudaMemcpyHostToDevice);
				host_temp[j] = node;
				//neighbors[i][j] = node;
				//printf(" %d %d ", i, j);
				j++;
				nodes = strtok(NULL, " ");
			}

			//Copy neighbor row into GPU for given node
			if(j == 2 * noNodes + 1) {
				cudaMemcpy(temp, host_temp, sizeof(int) * j, cudaMemcpyHostToDevice);
				cudaMemcpy(neighbors+i, &temp, sizeof(int*), cudaMemcpyHostToDevice); 
				j = 0; i++;
			}
		}
	}
	return neighbors;
}
