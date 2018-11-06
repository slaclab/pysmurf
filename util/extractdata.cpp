// extracts data from saved data files
// inputs: input_filename, output_filename,  first channel, last channel, averages, diagnostics

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
typedef uint32_t uint;


const uint buffsize = 10000000; // arbitrary for now.
const uint header_length = 128; // nubmer of bytes in header
const uint header_channel_offset = 4;  // bytes
const uint time_lower_offset = 72; // offset to 32 bit timing sysetm counter nanoseconds
const uint time_upper_offset = 76;  // offset to 32 bit timing system contuer seconds (maybe broken)
const uint mce_counter_offset = 96; 
const uint frame_counter_offset = 84; 
const uint data_size = 4; // data size bytes
const uint max_channels = 4096; // will really use 528

int main(int argc, char *argv[])
{
  char infilename[256];
  char outfilename[256];
  uint first_channel;
  uint last_channel;
  uint averages;
  uint read_size; // size of single read frame
  uint read_block; // number of bytes to read on each read operation.
  int fdin; // File descriptor input
  FILE  *fpout; // File pointer output
  uint8_t *buffer;
  uint32_t num_channels;  // from header, number of channels in data
  uint output_channels;  // number of channels to write
  uint read_frames; // number of frames to read each call
  uint proc_frames;
  uint bytes_read;
  uint8_t *bufptr; 
  int32_t data_out[max_channels]; // output data 
  uint32_t lower_time_counter; 
  uint32_t upper_time_counter;
  uint32_t initial_upper_time; 
  uint ctr; 
  uint tmp;
  uint avgcnt; // counter for average cycle
  uint outn; // output number
  double x, dtu, dtl;  // used for output, to simply syntax
  uint diagmode; 
  uint32_t mcecounter; 
  uint32_t framecounter; 

  buffer =(uint8_t*) malloc(buffsize * sizeof(uint8_t)); 
  strcpy(infilename, "data.dat");
  strcpy(outfilename, "outfile.dat");
  first_channel = 0;
  last_channel = 0;
  averages = 1;
  diagmode = 0;  // normal mode 
  
  if(argc > 1) strcpy(infilename, argv[1]);
  if(argc > 2) strcpy(outfilename, argv[2]);
  if(argc > 3)
    {
      first_channel = strtol(argv[3], NULL, 10);
      last_channel = first_channel;
    }
  if(argc > 4) last_channel = strtol(argv[4], NULL, 10);
  if(argc > 5) averages = strtol(argv[5], NULL, 10);
  if(argc > 6) diagmode = strtol(argv[6], NULL, 10); 
  if(!(fdin = open(infilename, O_RDONLY))) return(0);
  if(!(fpout = fopen(outfilename, "w"))) return(0);
  
  

  read(fdin, buffer, header_length); // read first header (will assume all ar the same
  close(fdin); // ugly way to rewind -must be a better way
  
  num_channels = *((uint32_t*)(buffer + header_channel_offset)); // sets read block size
  initial_upper_time  = *((uint32_t*)(buffer + time_upper_offset)); // 64 bit counter 


  //num_channels = 528; // TEST TEST TEST - REMOVE FOR PRODUCTION, UGLY KLUDGE 

  if(first_channel > num_channels) first_channel = num_channels;
  if(last_channel > num_channels) last_channel = num_channels;
  if (last_channel < first_channel) last_channel = first_channel; 
  output_channels = 1 + last_channel - first_channel;

  read_size = num_channels * data_size + header_length;  // all in bytes
  printf("arc = %d, fdin = %d, infile = %s, outfile = %s, numchans = %u\n", argc, fdin, infilename, outfilename, num_channels);
  
  tmp = buffsize / read_size / averages;
  read_frames = tmp * averages;  // number of frames at each read
  read_block = read_frames * read_size; // total bytes to read each tim
  

 
  
  if (!(fdin = open(infilename, O_RDONLY))) return(0); // now open again. (really dumb)
  outn = 0;  // which output sample are we on.
  for(ctr = 0; ctr < 1000000000; ctr++)           //MAIN READ LOOP
    { 
      bytes_read = read(fdin, buffer, read_block); // read-in block of data to process
      proc_frames = bytes_read / read_size; 
      printf("read block = %u, bytes read = %u , proc_frames = %u\n", ctr, bytes_read, proc_frames);
      avgcnt = 0;
      for (uint j = 0; j < proc_frames; j++)
	{
	  bufptr = buffer + j * read_size; // pointer to current buffer
	  
	  for(uint n = first_channel; n <= last_channel; n++)
	    {	
	      data_out[n - first_channel] += *((int32_t*)(buffer + header_length+4*n + read_size * j)); // sum data	
	    }
	  avgcnt++;  // increment average count
	  if (avgcnt == averages) // done averaging, time to write
	    {
	      upper_time_counter = *((uint32_t*)(buffer + time_upper_offset + read_size * j)); // 32 bit counter
	      lower_time_counter =  *((uint32_t*)(buffer + time_lower_offset + read_size * j)); 
	      dtl = (double) lower_time_counter; 
	      dtu = (double) (upper_time_counter - initial_upper_time);  // now a double
	      framecounter = *((uint32_t*)(buffer + frame_counter_offset + read_size * j)); 
	      mcecounter =  *((uint32_t*)(buffer + mce_counter_offset + read_size * j)); 
	      fprintf(fpout, "%12.6f ", dtu + dtl/1e9);
	      if(diagmode == 1)
		{
		  fprintf(fpout, "%10d %10d ", framecounter, mcecounter);
		}
	      for(uint n = first_channel; n <= last_channel; n++)
		{	
		  x =  data_out[n - first_channel]; // convert to float
		  fprintf(fpout, "%12.3f ", x / (double) averages);
		  data_out[n - first_channel] = 0;  // clear
		} 
	      fprintf(fpout, "\n"); // new line
	      avgcnt = 0;
	      outn = 0; 
	    }
	}
      if(bytes_read < read_block) break; // done
    }

  if (fdin) close(fdin);
  if (fpout) fclose(fpout);
  

}

