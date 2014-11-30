
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cstdio>

#define FRAMES 30
#define HEIGHT 320
#define WIDTH 480



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int ***allocateVideo( int x, int y, int t );
int ***initializeVideo(int ***video, int x, int y, int t);



__global__ void parrallelBlockMatching()
{

}

__global__ int ***motionEstimation(int ***video, int ***motion, int x, int y, int t, int b, int r)
{
	int f;
	for(f=0;f<t-1;f++)
	{
		motion=motionFrame(video,motion,x,y,f,b,r);
	}
	return motion;
}
__global__ int ***motionFrame(int ***video, int ***motion, int x, int y, int t, int b, int r)
{
	int i,j;
	
	for(i=r;i<=x-b-r;i++)
	{
		for(j=r;j<=y-b-r;j++)
		{
			motion[t][i][j]=blockMatching(video,i,j,t,b,r);
			//to optimize: make i and j increment by b instead of 1
		}
	}
	
	return motion;
}
__global__ int blockMatching(int ***video, int x, int y, int t, int b, int r)
{
	int diff, best_diff, i, j;
	double best_dist,dist;
	best_diff=1000000;
	
	for(i=x-r;i<=x+r;i++)
	{
		for(j=y-r;j<=y+r;j++)
		{
			diff=blockDiff(video,x,y,t,i,j,b);
			
			if(diff<best_diff)
			{
				best_diff=diff;
				best_dist=((x-i)*(x-i)+(y-j)*(y-j));
			}
			else if(diff==best_diff)
			{
				dist=((x-i)*(x-i)+(y-j)*(y-j));
				if(dist<best_dist)
				{
					best_dist=dist;
				}
			}
		}
	}
	
	return best_dist;
}
__global__ int blockDiff(int ***video, int x, int y, int t, int cx, int cy, int b)
{
	int i,j;
	int diff=0;
	
	for(i=0;i<b;i++)
	{
		for(j=0;j<b;j++)
		{
			diff+=abs(video[t][x+i][y+j]-video[t+1][cx+i][cy+j]);
			//to optimize: break when diff exceed best_diff
		}
	}
	
	return diff;
}

int main()
{
	// Allocate and Initialize video 
	int ***video = allocateVideo(WIDTH,HEIGHT,FRAMES);
	int ***motionVideo = allocateVideo(WIDTH,HEIGHT,FRAMES);

	video = initializeVideo(video,WIDTH,HEIGHT,FRAMES);

	// Allocate and initialize in device
	int *** d_video;
	int *** d_motionVideo;
	d_video = (int***) cudaMalloc(&d_video,sizeof(int)*FRAMES*WIDTH*HEIGHT);
	d_motionVideo = (int***) cudaMalloc(&d_motionVideo,sizeof(int)*FRAMES*WIDTH*HEIGHT);
	
	cudaMemcpy(d_video,&video,sizeof(int)*FRAMES*WIDTH*HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_motionVideo,&motionVideo,sizeof(int)*FRAMES*WIDTH*HEIGHT, cudaMemcpyHostToDevice);
	

	// perform serial computations in device
	motionEstimation <<< 1,1 >>> (d_video,d_motionVideo,WIDTH,HEIGHTFRAMES);
	// perform parallel computations in device 
	cudaFree(d_video);
	free(video);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.

int ***allocateVideo( int x, int y, int t )
{
  int i,j;
  int ***video;
  video = (int ***) malloc(t * sizeof(int**));
  for (i=0;i<t;i++)
  {
	video[i] = (int **) malloc( x * sizeof(int*));
	for (j=0;j<x;j++)
	  video[i][j] = (int *) malloc( y * sizeof(int));
  }
  return video;
}

int ***initializeVideo(int ***video, int x, int y, int t)
{
  int i,j,l,l_f;
  
  l=y/2;
  l_f=1;
  for (i=0;i<t;i++)
  {
    l_f=rand()%5-2;
	l+=l_f;
	if(l<2) l=1;
	if(l>y-4) l=y-4;
	
	for (j=1;j<x-1;j++)
	{
		video[i][j][l] = (j+15)%256;
		video[i][j][l+1] = (j+16)%256;
		video[i][j][l+2] = (j+15)%256;
	}
  }
  return video;
}


