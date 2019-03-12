#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int** arr)
{
	for (int i=0; i<3; i++)
		printf("%d\n", arr[i][0]);
}

int main()
{
	int arr[][3] = {{1},{2},{3}}; // 3 arrays, 1 element each
	char x = 0b100000100;
	printf("Bin: %d\n", x);

	int **d_arr;

	cudaMalloc((void***)(&d_arr), sizeof(int*)*3); // allocate for 3 int pointers

	for (int i=0; i<3; i++)
	{
		int* temp;
		cudaMalloc( (void**)  &(temp), sizeof(int) * 1 ); // allocate for 1 int in each int pointer
		cudaMemcpy(temp, arr[i], sizeof(int) * 1, cudaMemcpyHostToDevice); // copy data
		cudaMemcpy(d_arr+i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
	}

	kernel<<<1,1>>>(d_arr);
	cudaDeviceSynchronize();
	cudaDeviceReset();
}
