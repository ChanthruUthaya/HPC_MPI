
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>
#include "omp.h"

#define MASTER 0
#define NDIMS 2

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void getcols(int rank, int nprocs, int nx, int ny, int * cols, int * halosize, int * startcoord);
void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);
void halo_exchange(int halosize, int rank, int nprocs, float * colarea, int cols, MPI_Status status, int height);

int main(int argc, char* argv[])
{
  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  int nprocs, rank;
  //int reorder = 0;       /* an argument to MPI_Cart_create() */
  MPI_Status status;
  MPI_Comm cart_comm;
  //int dims[NDIMS]; /* array to hold dimensions of an NDIMS grid of processes */
  //int periods[NDIMS]; /* array to specificy periodic boundary conditions on each dimension */
  //int coords[NDIMS]; /* array to hold the grid coordinates for a rank */
  int tag = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

 // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);
  double maxTime = 0;

  if(nprocs == 1){

    // Call the stencil kernel
    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
      stencil(nx, ny, width, height, image, tmp_image);
      stencil(nx, ny, width, height, tmp_image, image);
    }
    double toc = wtime();

    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }
  else{

        int halo_size = 0;
        int ncols = 0;
        int startcoord = 0;
        getcols(rank, nprocs, nx, ny, &ncols, &halo_size, &startcoord);


        float* colarea = malloc(sizeof(float) * ncols * height);
        float* tmp_colarea = malloc(sizeof(float) * ncols * height);
        for(int i = 0; i < ncols-2; i++){
          for(int j = 0; j < height ; j++){
            colarea[j + height*(i+1)] = image[(startcoord + 1)*height + j + i*height];
          }
        }
        
        double tic = wtime();
        for (int t = 0; t < niters; ++t) {
            halo_exchange(halo_size, rank, nprocs, colarea, ncols, status, height);
            stencil(ncols - 2, ny, ncols, height, colarea, tmp_colarea);
            halo_exchange(halo_size, rank, nprocs, tmp_colarea, ncols, status, height);
            stencil(ncols - 2, ny, ncols, height, tmp_colarea, colarea);
        }
        double toc = wtime();

        if(rank == MASTER){
          for(int i = 0; i < ncols-2; i++){
            for(int j = 0; j < height ; j++){
              image[(startcoord + 1)*height + j + i*height] = colarea[j + height*(i+1)];
            }
          }
          int read_info[2] = {0,0};
          for(int i = 1; i < nprocs; i++){
              MPI_Recv(&read_info[0],2,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
              for(int j = 0; j < read_info[1]-2; j++){
                MPI_Recv(&image[((read_info[0]+1)*height) + j*height],height,MPI_FLOAT,i,tag, MPI_COMM_WORLD,&status);
              }
          }
           maxTime = toc-tic;
           double rTime = 0;
           for (int r = 1; r < nprocs; r++) {
               MPI_Recv(&rTime, BUFSIZ, MPI_DOUBLE, r, tag, MPI_COMM_WORLD, &status);
               if (rTime > maxTime) maxTime = rTime;
           }
            // Output
          printf("------------------------------------\n");
          printf(" runtime: %lf s\n", maxTime);
          printf("------------------------------------\n");
          //printf("done\n");

          output_image(OUTPUT_FILE, nx, ny, width, height, image);
        }
        else{
          int send_info[2] = {startcoord, ncols};
          MPI_Send(&send_info[0], 2, MPI_INT,0,tag, MPI_COMM_WORLD);
          for(int i = 0; i < ncols -2 ; i++){
              MPI_Send(&colarea[(i+1)*height], height, MPI_INT,0,tag, MPI_COMM_WORLD);
          }
          double time = toc-tic;
          MPI_Send(&time, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
  }

  free(image);
  free(tmp_image);

  MPI_Finalize();
}

void getcols(int rank, int nprocs, int nx, int ny, int * ncols, int * halosize, int * startcoord){
     int leftover = (ny) % nprocs;
     int divisible = (ny) - leftover;
     *ncols = divisible/nprocs;
     if(rank <leftover){
       *ncols += 1;
     }
     int cols = divisible/nprocs;
     *startcoord = rank*cols;
     if(rank < leftover){
       *startcoord += rank;
     }
     else{
       *startcoord += leftover;
     }
     *halosize = ny+2;
     *ncols += 2;
}

void halo_exchange(int halosize, int rank, int nprocs, float * colarea, int cols, MPI_Status status, int height){

    MPI_Sendrecv(&colarea[height], halosize, MPI_FLOAT, (rank -1 + nprocs) % nprocs, 0, &colarea[(cols-1)*height], halosize, MPI_FLOAT, (rank +1 + nprocs) % nprocs, 0, MPI_COMM_WORLD, &status);
    MPI_Sendrecv(&colarea[(cols-2)*height], halosize, MPI_FLOAT, (rank +1 + nprocs)%nprocs, 0, &colarea[0], halosize, MPI_FLOAT, (rank -1 + nprocs)%nprocs, 0, MPI_COMM_WORLD, &status);

    if(rank == 0){
       for(int j = 0; j < halosize; j++){
        colarea[j] = 0.0f;
      }
    }
    if(rank == nprocs -1){
       for(int j = 0; j < halosize; j++){
        colarea[j + (cols-1)*height] = 0.0f;
      }
    }
}

void stencil(const int nx, const int ny, const int width, const int height,
             float* restrict image, float* restrict tmp_image)
{
  //#pragma omp parallel for shared(image, tmp_image, nx, ny) schedule(dynamic, 10)
  for (int i = 1; i < nx + 1; ++i) {
    for (int j = 1; j < ny + 1; ++j) {
      tmp_image[j + i * height] = (image[j + i* height] * 0.6f) + (image[j + (i - 1) * height] * 0.1f) +(image[j+(i + 1) * height] * 0.1f) + (image[j - 1 + i* height] * 0.1f)+ (image[j + 1 + i* height] * 0.1f);
    }
  }
}
// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
//test
