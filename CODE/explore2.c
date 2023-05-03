/* This program solves the problem with the BDF method,
 * Newton iteration with the CVDENSE dense linear solver, 
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Header files with a description of contents used here */

#include <cvode/cvode.h>             /* prototypes for CVODE fcts. and consts. */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, functions, and macros */

/* User-defined vector and matrix accessor macros: Ith, IJth */

/* These macros are defined in order to write code which exactly matches
   the mathematical problem description given above.

   Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..MAX] */

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..MAX */


/* Problem Constants */

#define NEQ   16                /* number of equations  */
#define ATOL  RCONST(1.0e-6)   /* vector absolute tolerance components */
#define T0    RCONST(0.0)      /* initial time           */
#define T1    RCONST(0.1)      /* first output time      */
#define TMULT RCONST(0.1)     /* output time factor     */

#define N_TYPES 3
enum {PN,LNI,LNE};

typedef struct {
  double CA; // Axon capacitance on pico F
  double CS; // Soma capacitance on pico F
  double gal; // Leak axon on nano S
  double gsl; // Leak soma on nano S
  double gAS; // connecting conductance on nano S
  double gNa; // [Na] condutance on nano S
  double gKd; // delayed rectifier conductance on nano S
  double gCa; // [Ca] conductance on nano S
  double MU;  // [Ca] dynamics dissipation (unitless)
  double gKCa; // [K]([Ca]) conductance on nano S
  double gA;   // concutance for IA on nano S
  double glow; // [Ca] low threshold conductance (nano S)
  double gh;   // Low threshold curren (nano S)
  double kfGABAA; // rise constant for GABA A on ms-1
  double krGABAA; // decay constant for GABA A on ms-1
  double kfAMPA;  // rise constant for AMPA on ms-1
  double krAMPA; // decay constant for AMPA on ms -1
  double kfPN; // rise constant for PN on ms-1
  double krPN; // decay constant for PN on ms -1

  double Vt;    // Spike activation threshold
  
  //NEW GABA parameter from Ermentrout and Terman: Mathematical Foundations of Neuroscience
  double ar;
  double br;
  double ad;
  double K3;   /* ms -1 */
  double K4;  /* ms -1 */
  // This for the GABA-B current
  double Kd;     /* dimensionless for a Hill function */

} model_parameters;

#define MAXSPIKES 100

typedef struct {
  int top;
  double ts[MAXSPIKES];
  int flag;
} queue;

typedef struct {
  int tipo;
  int Nampa; // Number of incoming AMPA conections;
  double *gampa;
  int *Iampa;

  int NgabaA; // Number of incoming GABA connections
  double *ggabaA;
  int *Igaba;

  int NgabaB; // Number of incoming GABA connections
  double *ggabaB;
  int *IgabaB;

  double gglomeruli;
  int   Ig;
  double shift;
  queue spikes;
} model_type;


typedef struct {
  int Ntotal;
  int Nlocal[N_TYPES];
  int Ng; // Number of glomeruli
  double pactive;  // Probability of having an active glomeruli
  double Gg[N_TYPES]; // Strengh fro the receptors
// NUmber of active glomeruli
  int Nag;
  int *activeG;
  double p[N_TYPES][N_TYPES+1]; // Probability of connections
  double s[N_TYPES][N_TYPES+1]; // Synaptic strenght
  double pg[N_TYPES];  // prob. of connection in glomeruli
  model_type *neuron;
  double scale;
  double rate;
  model_parameters param[N_TYPES];
 
} nervous_system;

/* Functions Called by the Solver */

static int f(double t, N_Vector y, N_Vector ydot, void *user_data);

/* Private functions to output results */

static void PrintOutput(FILE *,double , N_Vector,nervous_system *);
static void PrintAverage(FILE *,double , N_Vector ,nervous_system *,double,double,double,double) ;
static void PrintOutputCa(FILE *,double , N_Vector,nervous_system *);

/* Private function to print final statistics */

static void PrintFinalStats(void *cvode_mem);

/* Private function to check function return values */

static int check_flag(void *flagvalue, char *funcname, int opt);

void set_initial_conditions(N_Vector,int,nervous_system * );

double freq(int,nervous_system *,double);

void Save_Freq(FILE *, double,nervous_system *);

void save_final_conditions(N_Vector,nervous_system *);

#include "rgenerator.h"

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */
#define PARAMETERS 4
#define IDC       1
#define DATAFILE  2
#define SCALE     3 
#define RATE      4

double Idc;
double tmp_gaba;
double peak_gabaA;
double peak_gabaB;
double local_gabaA;
double local_gabaB;
double PNlocal_gabaA;
double PNlocal_gabaB;
double LNlocal_gabaA;
double LNlocal_gabaB;

void set_parameters(nervous_system *);
void read_network(char *, nervous_system *);
void release_neurons(nervous_system *);

int main(int argc, char **argv)
{
  double tolerance=1e-4;
  double alpha=2.0/(1.0+1000.0);
  double t, tout;
  N_Vector y=NULL;
  void *cvode_mem=NULL;
  int flag;
  double tmax;
  nervous_system NS;
  FILE *ar,*br,*cr,*dr;
  double PNmeanGA=0;
  double PNmeanGB=0;
  double LNmeanGA=0;
  double LNmeanGB=0;
  peak_gabaA=0.0;
  peak_gabaB=0.0;
  

  if ((argc-1)!=PARAMETERS) {
    puts("Current");				
    puts("Input data file");
    puts("scale inhibition");
    return -1;
  }
  NS.scale=atof(argv[SCALE]);
  NS.rate=atof(argv[RATE]);
  Idc=atof(argv[IDC]);
  tmax=9*1000.0;

  read_network(argv[DATAFILE],&NS);
  set_parameters(&NS);
  char name1[100],name2[100],name3[100],name4[100];
  sprintf(name1,"time_series_%4.4lf_r%4.2lf_%4.2lf_%s",NS.scale,NS.rate,Idc,argv[DATAFILE]);
  puts(name1);
  sprintf(name2,"average_activity_%4.4lf_r%4.2lf_%4.2lf_%s",NS.scale,NS.rate,Idc,argv[DATAFILE]);
  sprintf(name3,"freq_%4.4lf_r%4.2lf_%4.2lf_%s",NS.scale,NS.rate,Idc,argv[DATAFILE]);
  sprintf(name4,"Ca_%4.4lf_r%4.2lf_%4.2lf_%s",NS.scale,NS.rate,Idc,argv[DATAFILE]);
  ar=fopen(name1,"w");
  if (ar==NULL) {puts("Cannot open output file");exit(0);}
  br=fopen(name2,"w");
  if (br==NULL) {puts("Cannot open output file");exit(0);}
  cr=fopen(name3,"w");
  if (cr==NULL) {puts("Cannot open output file");exit(0);}
  dr=fopen(name4,"w");
  if (dr==NULL) {puts("Cannot open output file");exit(0);}
  /* Create serial vector of length NEQ for I.C. */
  printf("Total # of ODEs is %d \n",NEQ*NS.Ntotal);
  y = N_VNew_Serial(NEQ*NS.Ntotal);
  if (check_flag((void *)y, "N_VNew_Serial", 0)) return(1);

  /* Initialize y */
  set_initial_conditions(y,NS.Ntotal,&NS);

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);
  
  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, f, T0, y);
  if (check_flag(&flag, "CVodeInit", 1)) return(1);

  flag = CVodeSStolerances(cvode_mem, tolerance, ATOL);

  flag = CVodeSetUserData(cvode_mem, (void *) (&NS)); 
  
  if (flag!=CV_SUCCESS) {
    printf("Something wrong setting the pointer to the data");
    exit(0);
  }
  /* Call CVDense to specify the CVDENSE dense linear solver */
  flag = CVDense(cvode_mem, NEQ*NS.Ntotal);
  if (check_flag(&flag, "CVDense", 1)) return(1);
  /* Set the Jacobian routine to internal estimation */
  flag=CVDlsSetDenseJacFn(cvode_mem, NULL);
  
  // flag = CVDlsSetDenseJacFn(cvode_mem, Jac);
  // if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)) return(1);

  /* In loop, call CVode, print results, and test for error.
     Break out of loop when NOUT preset output times have been reached.  */

  puts("Start ODE integration");
  tout = T1;
  while(1) {
    flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    
    if (check_flag(&flag, "CVode", 1)) break;
    if (flag == CV_SUCCESS) {
	if (tout>=2000) {
	PrintOutput(ar,t-3500,y,&NS );fflush(ar);
	PrintAverage(br,t-3500, y,&NS,PNmeanGA,PNmeanGB,LNmeanGA,LNmeanGB);fflush(br);
	Save_Freq(cr,t-3500,&NS);fflush(cr);
	PrintOutputCa(dr,t-3500,y,&NS );fflush(dr);
	}
	if ((((int)t)%1000)==0) {printf("Time %lf ms\n",t);fflush(stdout);}
	tout += TMULT;
	PNmeanGA=alpha*PNlocal_gabaA+(1-alpha)*PNmeanGA;
	PNmeanGB=alpha*PNlocal_gabaB+(1-alpha)*PNmeanGB;
	LNmeanGA=alpha*LNlocal_gabaA+(1-alpha)*LNmeanGA;
	LNmeanGB=alpha*LNlocal_gabaB+(1-alpha)*LNmeanGB;
	//printf("%lg %lg\n",tout,TMULT);
    }

    if (tout>tmax) break;
  }

  save_final_conditions(y,&NS);

  /* Free y vector */
  N_VDestroy_Serial(y);

  /* Free integrator memory */
  CVodeFree(&cvode_mem);

  /* Relase the network */
  release_neurons(&NS); 
  fclose(cr);
  fclose(br);
  fclose(ar);
  fclose(dr);

  return(0);
}

double freq(int i,  nervous_system *NS,double ti) {
 int ini=NS->neuron[i].spikes.top;

 --ini;
 if (ini<0) ini=MAXSPIKES+ini;
 // double ti=NS->neuron[i].spikes.ts[ini];
 double t0=ti-200.0;
 if (ti<0) return 0.0;
 if (t0<0) return 0.0;

 int nt=0;
 double tnext;
 //puts("********************");
 do {
   
   --ini;
   if (ini<0) ini=MAXSPIKES+ini;
   tnext=NS->neuron[i].spikes.ts[ini];
   if (tnext<0) return 0.0;
   if (tnext<t0) return (double)nt/1.0;
   //printf("%d %lf (%lg,%lg)\n",i,ti-tnext,t0,tnext);

   ++nt;
   ti=tnext;

 } while (tnext>t0);

 return ((double)nt)*5.0; 
}


void Save_Freq(FILE *br, double t,nervous_system *NS) {
 int i=0;
 int ni=0;
  double sum=0;
  fprintf(br,"%lg\t",t);
  ni=0.0;
  for(i=0;i<NS->Nlocal[0];++i) {
    sum+=freq(i,NS,t);
    ++ni;
  }
  sum/=(double) ni;
  fprintf(br,"%lg ",sum);
  /*  sum=0;
  for(;i<(NS->Nlocal[0]+NS->Nlocal[1]);++i)
    sum+=freq(i,NS);;
  sum/=(double) NS->Nlocal[1];
  fprintf(br,"%lg ",sum);*/
  sum=0;
  ni=0;
  for(i=NS->Nlocal[1];i<(NS->Nlocal[0]+NS->Nlocal[1]);++i) {
    sum+=freq(i,NS,t);
    ++ni;
  }
  sum/=(double) ni;

  double std=0.0;

  for(i=NS->Nlocal[1];i<(NS->Nlocal[1]+NS->Nlocal[2]);++i)
    std+=(freq(i,NS,t)-sum)*(freq(i,NS,t)-sum);

  std/=(double) NS->Nlocal[1];
  std=sqrt(std);

  fprintf(br,"%lg\n",sum,std);



}

void read_network(char *name, nervous_system *NS) {
  FILE *ar;
  char *b;
  char tmp[200];
  int e;
  ar=fopen(name,"r");

  if (ar==NULL) {
    printf("input network file not present %s\n",name);
    exit(0);
  }

    b=fgets(tmp,200,ar);
    if (b==NULL) {
      puts("No PN # of neurons");
      exit(0);
    }
    NS->Nlocal[0]=atoi(tmp);

    b=fgets(tmp,200,ar);
    if (b==NULL) {
      puts("No LNI # of neurons");
      exit(0);
    }
    NS->Nlocal[1]=atoi(tmp);

    b=fgets(tmp,200,ar);
    if (b==NULL) {
      puts("No LNE # of neurons");
      exit(0);
    }
    NS->Nlocal[2]=atoi(tmp);

    b=fgets(tmp,200,ar);
    if (b==NULL) {
      puts("No glomeruli # of neurons");
      exit(0);
    }
    NS->Ng=atoi(tmp);

    b=fgets(tmp,200,ar);
    if (b==NULL) {
      puts("Probability of getting a glomeruli active");
      exit(0);
    }
    NS->pactive=atof(tmp);
    printf("Prob. of getting an actvie glo. %lf\n",NS->pactive);

    e=fscanf(ar,"%lf %lf %lf",&(NS->Gg[0]),&(NS->Gg[1]),&(NS->Gg[2]));
   if (e!=3) {puts("Wrong input strenght to glo");exit(0);}
   
    NS->Ntotal=NS->Nlocal[0]+NS->Nlocal[1]+NS->Nlocal[2];
    printf("Ntotal %d, PN %d, LNI %d, LNE %d\n", NS->Ntotal,NS->Nlocal[0],NS->Nlocal[1],NS->Nlocal[2]);
    e=fscanf(ar,"%lf %lf %lf",&(NS->pg[0]),&(NS->pg[1]),&(NS->pg[2]));
    if (e!=3) {puts("Wrong input format for connection to glo");exit(0);}
    

    e=fscanf(ar,"%lf %lf %lf %lf",&NS->s[0][0],&NS->s[0][1],&NS->s[0][2],&NS->s[0][3]);
    if (e!=4) {puts("Wrong input format strenght 0");exit(0);}
    
    e=fscanf(ar,"%lf %lf %lf %lf",&NS->s[1][0],&NS->s[1][1],&NS->s[1][2],&NS->s[1][3]);
    if (e!=4) {puts("Wrong input format strenght 1");exit(0);}
    printf,


    e=fscanf(ar,"%lf %lf %lf %lf",&NS->s[2][0],&NS->s[2][1],&NS->s[2][2],&NS->s[2][3]);
    if (e!=4) {puts("Wrong input format strengh 2");exit(0);}
    printf("Synaptic strenght of connections\n%lg %lg %lg\n%lg %lg %lg\n%lg %lg %lg\n",NS->s[0][0],NS->s[0][1],NS->s[0][2],NS->s[1][0],NS->s[1][1],NS->s[1][2],NS->s[2][0],NS->s[2][1],NS->s[2][2],NS->s[2][2]);

    fclose(ar);

    NS->neuron=(model_type *)malloc(sizeof(model_type)*NS->Ntotal);
    if (NS->neuron==NULL) {
      puts("There is no space for creating neurons");
      exit(0);

    }

    int i=0;
    for(i=0;i<NS->Ntotal;++i) {
      if (i<NS->Nlocal[0]) {
	NS->neuron[i].tipo=0;
      } else if (i<(NS->Nlocal[1]+NS->Nlocal[0])) {
	NS->neuron[i].tipo=1;
      } else {
	NS->neuron[i].tipo=2;
      }
      NS->neuron[i].Nampa=0;
      NS->neuron[i].NgabaA=0;
      NS->neuron[i].NgabaB=0;
      NS->neuron[i].shift=(genrand()-0.5);
      NS->neuron[i].spikes.top=0;
      NS->neuron[i].spikes.flag=0;
      NS->neuron[i].spikes.ts[MAXSPIKES-1]=-1.0;
    }

    int index[NS->Ntotal];
    int nl=0;
    int j,k,k2;
    int iglo[NS->Ntotal][NS->Ng];
    int niglo[NS->Ntotal];
    int gpn=0;
    int nln=0;
    // Set the connections to the glomeruli
    for(i=0;i<NS->Ntotal;++i) {
      if (NS->neuron[i].tipo==PN) {
	// Set up the ORC input
	int s=(gpn%(NS->Ng));
	++gpn;
	NS->neuron[i].Ig=s;
	NS->neuron[i].gglomeruli=NS->Gg[PN];
	printf("PN neuron %d is connected to glomeruli %d %lf\n", i,s,NS->neuron[i].gglomeruli);fflush(stdout);
	niglo[i]=1;
	iglo[i][0]=s;
      } else if (NS->neuron[i].tipo==LNI) {
	// Set up the ORC input
	int s=(nln%(NS->Ng));//(int)(genrand()*(double)NS->Ng);
	++nln;
	NS->neuron[i].Ig=s;
	NS->neuron[i].gglomeruli=NS->Gg[LNI];

	niglo[i]=0;
	for(j=0;j<NS->Ng;++j) 
	  if (genrand()<NS->pg[LNI]) {
	    iglo[i][niglo[i]]=j;
	    ++niglo[i];
	  }
	printf("LNI(%d)<->",i);
	for(j=0;j<niglo[i];++j) 
	  printf("%d ",iglo[i][j]);
	printf("\n");
	
      } else if (NS->neuron[i].tipo==LNE) {
	// Set up the ORC input
	int s=(int)(genrand()*(double)NS->Ng);
	NS->neuron[i].Ig=s;
	NS->neuron[i].gglomeruli=0.0;

	
      }
    }
    // Test connections
    for(i=0;i<NS->Ntotal;++i) {
      printf("Neuron %d \n",i);fflush(stdout);
      if (NS->neuron[i].tipo==PN) {
	
	// Set up the excitatory connexctions to the PNs
	int nce=0;
	int conn[NS->Ntotal*NS->Ntotal];
		/*	for(j=NS->Nlocal[0]+NS->Nlocal[1];j<NS->Ntotal;++j) {
	  for(k=0;k<niglo[j];++k) {
	    if (iglo[j][k]==iglo[i][0]) { 
	      conn[nce]=j;
	      ++nce;
	    }
	  }
	}
	printf("PN(%d) receives %d LNE\n",i,nce);fflush(stdout);*/
	NS->neuron[i].Nampa=nce;
	/*	NS->neuron[i].gampa=(double *)malloc(sizeof(double)*nce);
	NS->neuron[i].Iampa=(int *)malloc(sizeof(int)*nce);
	for(j=0;j<nce;++j) {
	  NS->neuron[i].gampa[j]=NS->s[0][2]*genrand();
	  NS->neuron[i].Iampa[j]=conn[j];	    
	  }*/

	
	


	// Set up the inhibitory connexctions to the PNs
	nce=0.0;
	for(j=NS->Nlocal[0];j<NS->Nlocal[0]+NS->Nlocal[1];++j) {
	// does not receive inhibition from the same glomerulus
	  if (NS->neuron[j].Ig!=NS->neuron[i].Ig) {
	    for(k=0;k<niglo[j];++k) {
	      if (iglo[j][k]==iglo[i][0]) { 
		conn[nce]=j;
		++nce;
	      }
	    }
	  }
	}
	printf("PN(%d) receives %d LNI <--",i,nce);fflush(stdout);
	NS->neuron[i].NgabaA=nce;
	NS->neuron[i].ggabaA=(double *)malloc(sizeof(double)*nce);
	NS->neuron[i].Igaba=(int *)malloc(sizeof(int)*nce);
	for(j=0;j<nce;++j) {
	  NS->neuron[i].ggabaA[j]=NS->s[0][1]*NS->scale;
	  NS->neuron[i].Igaba[j]=conn[j];
	  printf("%d ",conn[j]);
	}
	printf("\n");
	// Let us assume here that there are addtional GABAB connections 
	NS->neuron[i].NgabaB=nce;
	NS->neuron[i].ggabaB=(double *)malloc(sizeof(double)*nce);
	NS->neuron[i].IgabaB=(int *)malloc(sizeof(int)*nce);
	printf("Connections strenght %lf\n",NS->s[0][2]);
	for(j=0;j<nce;++j) {
	  NS->neuron[i].ggabaB[j]=NS->s[0][2]; // the third column is GABAB strengh now
	  //printf("Connections strenght %lf %lf\n",NS->s[0][2],NS->neuron[i].ggabaB[j]);
	  NS->neuron[i].IgabaB[j]=conn[j];
	  printf("{%d %lf}",conn[j],NS->neuron[i].ggabaB[j]);
	}
	printf("\n");

      } else if (NS->neuron[i].tipo==LNI) {
	// Set up the excitatory connexctions to the LNIs	
	int nce=0;
	double conn[NS->Ntotal*NS->Ntotal];
	for(j=0;j<NS->Nlocal[0];++j) {
	  if (NS->neuron[i].Ig==NS->neuron[j].Ig) {
		conn[nce]=j;
		++nce;
	  }
	}
	  
	
	printf("LNI(%d) receives %d Lateral PN excitation\n",i,nce);fflush(stdout);

	NS->neuron[i].Nampa=nce;
	NS->neuron[i].gampa=(double *)malloc(sizeof(double)*nce);
	NS->neuron[i].Iampa=(int *)malloc(sizeof(int)*nce);
	for(j=0;j<nce;++j) {

	  NS->neuron[i].gampa[j]=NS->s[1][0];
	  NS->neuron[i].Iampa[j]=conn[j];	    
	  //printf("(%d,%f)",NS->neuron[i].Iampa[j],NS->neuron[i].gampa[j]);
	   
	  }
       
	// Set up the inhibitory connexctions to the LNIs
	nce=0;
	for(j=NS->Nlocal[0];j<NS->Nlocal[0]+NS->Nlocal[1];++j) {
	  //printf("%d (%d,%d)",j,niglo[i],niglo[j]);fflush(stdout);
	  if (i==j) continue;
	  for(k=0;k<niglo[j];++k) {
	      if (NS->neuron[i].Ig==iglo[j][k]) {
		  conn[nce]=j;
		  ++nce;
	      }
	  }
	}
	
	printf("LNI(%d) receives %d LNI\n",i,nce);fflush(stdout);
	NS->neuron[i].NgabaA=nce;
	if (nce>0) {
	  NS->neuron[i].ggabaA=(double *)malloc(sizeof(double)*nce);
	  NS->neuron[i].Igaba=(int *)malloc(sizeof(int)*nce);
	  for(j=0;j<nce;++j) {
	    NS->neuron[i].ggabaA[j]=NS->s[1][1]*NS->scale;
	    NS->neuron[i].Igaba[j]=conn[j];	    
	  }
	}
	// We assume that the dendritic tree is the same for GABA-A GABA-B
	NS->neuron[i].NgabaB=nce;
	if (nce>0) {
	  NS->neuron[i].ggabaB=(double *)malloc(sizeof(double)*nce);
	  NS->neuron[i].IgabaB=(int *)malloc(sizeof(int)*nce);
	  for(j=0;j<nce;++j) {
	    NS->neuron[i].ggabaB[j]=NS->s[1][2]*NS->scale; // We now place the strengh tof the connections in the third column
	    NS->neuron[i].IgabaB[j]=conn[j];	    
	  }
	}
      } else if (NS->neuron[i].tipo==LNE) {
	// Set up the excitatory connexctions to the LNEs	
	int nce=0;
	double conn[NS->Ntotal*NS->Ntotal];

	for(j=0;j<NS->Nlocal[0];++j) {
		conn[nce]=j;
		++nce;
	      }
	    
	nce=0;
	
	printf("LNE(%d) receives %d PN\n",i,nce);fflush(stdout);
	NS->neuron[i].Nampa=nce;
	NS->neuron[i].gampa=(double *)malloc(sizeof(double)*nce);
	NS->neuron[i].Iampa=(int *)malloc(sizeof(int)*nce);
	for(j=0;j<nce;++j) {
	  NS->neuron[i].gampa[j]=NS->s[2][0];
	  
	  NS->neuron[i].Iampa[j]=conn[j];
	  // printf("(%d,%f)",NS->neuron[i].gampa[j],NS->neuron[i].Iampa[j]);
	}
		

      }
    }
}

void release_neurons(nervous_system *NS) {
  int i;

  for(i=0;i<NS->Ntotal;++i) {
    if (NS->neuron[i].Nampa>0) {
      free(NS->neuron[i].gampa);
      free(NS->neuron[i].Iampa);
    }
    if (NS->neuron[i].NgabaA>0) {
      free(NS->neuron[i].ggabaA);
      free(NS->neuron[i].Igaba);
    }
    if (NS->neuron[i].NgabaB>0) {
      free(NS->neuron[i].ggabaB);
      free(NS->neuron[i].IgabaB);
    }
  }
  free(NS->activeG);
  free(NS->neuron);
}


void set_parameters(nervous_system *NS) {
  int tipo;
  //set up the active glomeruli
  int i;
  {
    printf("N glomeruli %d\n",NS->Ng);
    int cn[NS->Ng];
    int ag=0;
    for(i=0;i<NS->Ng;++i) {
      if (genrand()<NS->pactive) {
	cn[ag]=i;
	++ag;
      }
    }
    NS->Nag=ag;
    NS->activeG=(int *)malloc(sizeof(int)*NS->Nag);
    for(i=0;i<ag;++i)
      NS->activeG[i]=cn[i];
  }

  printf("Active G is %d\n",NS->Nag);

  {
    // LNI
    tipo=1;
    NS->param[LNI].CA= 10.0; /* pico F */
    NS->param[LNI].CS = 10.0; /* pico F */
    NS->param[LNI].gal=  1.6; /* nano S */
    NS->param[LNI].gsl=  1.6; /* nano S */
    NS->param[LNI].gAS= 10.0;  /* nano S */ 
    NS->param[LNI].gNa= 260.0;   /* nano S */
    NS->param[LNI].gKd= 80.0;  /* nano S */
    NS->param[LNI].gCa = 8.8 ;  /* nano S */
    NS->param[LNI].MU  =1.5 ;   /* Calcium dynamics dissipation*/
    NS->param[LNI].gKCa =1.5 ; /* nano S */
    //NS->param[LNI].gh  = 1.2   /* nano S */
    NS->param[LNI].gh  = 0.0;   /* nano S */
    NS->param[LNI].gA   =200.0; /* nano S */
    NS->param[LNI].kfGABAA =10.0; /* ms -1 */
    NS->param[LNI].krGABAA =0.1; /* ms -1 */    
    NS->param[LNI].Vt=-51.7;
    //Let's define GABA-B parameters for the neurotransmitter release
    //NS->param[LNI].ar= 1.8; /* mM-1 ms -1 */
    // NS->param[LNI].ar= 1.0; /* mM-1 ms -1 */
    NS->param[LNI].ar= 0.6; /* mM-1 ms -1 */
    //NS->param[LNI].br= 0.0004; /* mM-1 ms -1 */
    NS->param[LNI].br= 0.001;
    //NS->param[LNI].ad= 0.0012; /* ms -1*/
    //NS->param[LNI].K3=0.18/2.55;   /* ms -1 */
    NS->param[LNI].K3=0.3; 
    // NS->param[LNI].K4=0.005;  /* ms -1 */
    NS->param[LNI].K4=0.0025;
    // This for the GABA-B current
    NS->param[LNI].Kd=100;     /* dimensionless */
  } 
  {
    // LNE
    tipo=2;

    NS->param[tipo].CA=  10.0; /* pico F */
    NS->param[tipo].CS=  10.0; /* pico F */
    NS->param[tipo].gal=  1.6; /* nano S */
    NS->param[tipo].gsl=  1.6; /* nano S */
    NS->param[tipo].gAS= 65.0;  /* nano S */ 
    NS->param[tipo].gNa= 260.0;   /* nano S */
    NS->param[tipo].gKd= 80.0;   /* nano S */
    NS->param[tipo].gCa= 8.8;   /* nano S */
    NS->param[tipo].MU=  1.6;    /* Calcium dynamics dissipation*/
    NS->param[tipo].gKCa= 0.5;  /* nano S */
    NS->param[tipo].gh=   0.0;   /* nano S */
    NS->param[tipo].gA=   200.0; /* nano S */
    NS->param[tipo].kfAMPA= 10.0; /* ms -1 */
    NS->param[tipo].krAMPA= 1.0; /* ms -1 */
    NS->param[tipo].Vt=-52.1; /* Threshold for spike activation */

    NS->param[tipo].kfPN= 100.0; /* ms -1 */
    NS->param[tipo].krPN= 1.0; /* ms -1 */

    // This for the GABA-B current
    NS->param[tipo].Kd=100;     /* dimensionless */
  } 
  {
    // PN
    tipo=0;
    NS->param[tipo].CA=  10.0; /* pico F */
    NS->param[tipo].CS=  10.0; /* pico F */
    NS->param[tipo].gal=  1.6; /* nano S */
    NS->param[tipo].gsl=  1.6; /* nano S */
    NS->param[tipo].gAS= 65.0;  /* nano S */ 
    NS->param[tipo].gNa= 260.0;   /* nano S */
    NS->param[tipo].gKd= 80.0;   /* nano S */
    NS->param[tipo].gCa= 8.8;   /* nano S */
    NS->param[tipo].MU=  1.6;    /* Calcium dynamics dissipation*/
    NS->param[tipo].gKCa= 0.5;  /* nano S */
    NS->param[tipo].gh=   0.0;   /* nano S */
    NS->param[tipo].gA=   200.0; /* nano S */
    NS->param[tipo].kfAMPA= 100.0; /* ms -1 */
    NS->param[tipo].krAMPA= 1.0; /* ms -1 */
    NS->param[tipo].Vt=-52.1; /* Threshold for spike activation */

    NS->param[tipo].kfPN= 100.0; /* ms -1 */
    NS->param[tipo].krPN= 1.0; /* ms -1 */
    // This for the GABA-B current
    NS->param[tipo].Kd=100;     /* dimensionless */
  }

}


/* Here one writes the stimulation protocol */

double protocol(double t) {
  
    //if ((t>50.0)&&(t<=60)) return 100.0;
    //if ((t>60.0)&&(t<=200)) return -50.0;

    //if ((t>=200)&&(t<1200)) return Idc;
  if ((t>6200)&&(t<=7200)) return Idc;
  if ((t>12000)&&(t<=13000)) return Idc;

  return 0.0;
}
double protocolG(double t,int j,void *user_data) {
  //printf("S:%lf A:%d ID:%d\n", ((nervous_system *)user_data)->neuron[j].gglomeruli,((nervous_system *)user_data)->activeG,((nervous_system *)user_data)->neuron[j].Ig);
  int i;
  
  for(i=0;i<((nervous_system *)user_data)->Nag;++i) {
    if (((nervous_system *)user_data)->activeG[i]==((nervous_system *)user_data)->neuron[j].Ig) {
      if ((t>50.0)&&(t<=460)) return 20.0;
      if ((t>500.0)&&(t<=600)) return -30.0;
  
      if ((t>3500)&&(t<=7500)) return ((nervous_system *)user_data)->neuron[j].gglomeruli*Idc*exp(-(((nervous_system *)user_data)->rate)*(t-3500)/1000.0);
      
      //  if ((t>12000)&&(t<=13000)) return ((nervous_system *)user_data)->neuron[j].gglomeruli*Idc;
    }
  }
  return 0.0;
}
#define RTF 24.42002442
#define ABS( X ) (((X)>0.0)?(X):-(X))
#define Cagam(X,Y,Z) (1.0/(1.0+exp(((X)-(Y))/(Z)))) 
#define RECTI(X,S) ((fabs((X-S))>0.01)?((X-S)/(1.0-exp(2.0*(X-S)/RTF))):((-RTF+(X-S))*0.5))
#define VAMPA (0.0) /*mV*/
#define VGABA (-90.0) /*mV*/
#define gch2  0.2

double protocol2(double t) {
  double tt;
  double tmin=50000;
  double tmax=55000;
  double events=100.0; /* 10 Hz stimulation */
  int i;
  if (t<tmin) return 0;
  if (t>tmax) return 0;

  double sum=0.0;
  for(tt=tmin;tt<=t;tt+=events) {
    sum=Cagam(t,tt,-0.1)*Cagam(t,tt+events,0.1);
    // if ((t>tt)&&(t<=(tt+Ds))) {
    //  printf("sum %lf\n",sum);
    //}
  }

  return sum;
}

/*double protocol2(double t) {
  

  if ((t>50000)&&(t<=(50000+Idc))) return 1;

  return 0.0;
  }*/
double protocolLNE(double t) {
  
//    return 0.0;
    //if ((t>50.0)&&(t<=60)) return 100.0;
    //if ((t>60.0)&&(t<=200)) return -50.0;

    //if ((t>=200)&&(t<1200)) return Idc;
    //return 20.0;
  if ((t>6200)&&(t<=6700)) return Idc;
  if ((t>12000)&&(t<=13000)) return Idc;

  return 0.0;
}


/*
 * f routine. Compute function f(t,y). 
 */

static double synaptic_input(N_Vector y,void *user_data,int i/*neuron index*/,int j /*reference: NEQ*i*/) {
  int tipo=(((nervous_system *)user_data)->neuron[i].tipo);
  double s,kd;
  double   Isyn=0.0,Isyn2=0.0,Isyn3=0.0;;
  double factorPN=1.0/(double)(((nervous_system *)user_data)->Nlocal[0]);
  double factorLNI=1.0/(double)(((nervous_system *)user_data)->Nlocal[1]);
  double factorLNE=1.0/(double)(((nervous_system *)user_data)->Nlocal[2]);
  int k;

  //Integrate GABA-A
  //    printf("PN %d\n",((nervous_system *)user_data)->neuron[i].NgabaA);
  for(k=0;k<((nervous_system *)user_data)->neuron[i].NgabaA;++k)  {
    int ik=((nervous_system *)user_data)->neuron[i].Igaba[k];
    //printf("Relased %d %lg \n",ik,Ith(y,13+NEQ*ik));
    Isyn+=((nervous_system *)user_data)->neuron[i].ggabaA[k]*(Ith(y,2+j)/*Vs*/-VGABA)*Ith(y,14+NEQ*ik);
    //if (Ith(y,13+NEQ*ik)>0) 
    //printf("Relased %d %lg \n",ik,Ith(y,13+NEQ*ik));
  }
  local_gabaA=fabs(Isyn)/(double)((nervous_system *)user_data)->neuron[i].NgabaA;
  if (fabs(Isyn)>peak_gabaA) peak_gabaA=Isyn;
  //Intergate GABA-B
  for(k=0;k<((nervous_system *)user_data)->neuron[i].NgabaB;++k)  {
    int ik=((nervous_system *)user_data)->neuron[i].IgabaB[k];
//	printf("Relased %d %lg %lg\n",ik,Ith(y,16+NEQ*ik),Ith(y,15+NEQ*ik));//borre la mencion
    s=Ith(y, 16 +NEQ*ik); //Attention this has to be s of GABA-B
    kd=((nervous_system *)user_data)->param[tipo].Kd;
    Isyn2+=((nervous_system *)user_data)->neuron[i].ggabaB[k]*(Ith(y,2+j)/*Vs*/-VGABA)*(s*s*s*s/(s*s*s*s+kd*kd*kd*kd));
    //if (Ith(y,13+NEQ*ik)>0) 
    //printf("Relased %d %lg \n",ik,Ith(y,13+NEQ*ik));
  }
  local_gabaB=fabs(Isyn2)/(double)((nervous_system *)user_data)->neuron[i].NgabaB;
if (fabs(Isyn2)>peak_gabaB) peak_gabaB=Isyn2;
  
  //Isyn*=factorLNI;
  //  if (((nervous_system *)user_data)->neuron[i].NgabaA>0)
  //    Isyn/=(double)((nervous_system *)user_data)->neuron[i].NgabaA;
  //      puts("2");
  for(k=0;k<((nervous_system *)user_data)->neuron[i].Nampa;++k)  {
    int ik=((nervous_system *)user_data)->neuron[i].Iampa[k];
    Isyn3+=((nervous_system *)user_data)->neuron[i].gampa[k]*(Ith(y,2+j)/*Vs*/-VAMPA)*Ith(y,13+NEQ*ik);
  }
  //Isyn2*=factorPN;
    //if (Ith(y,13+NEQ*ik)>0.01) 
	//printf("in %d Relased %d %lg ->%lg \n",i,ik,Ith(y,13+NEQ*ik),((nervous_system *)user_data)->neuron[i].gampa[k]*(Ith(y,2+j)/*Vs*/-VAMPA)*Ith(y,13+NEQ*ik));
    //********************************************
    //PN
    //******************************************


  return Isyn+Isyn2+Isyn3;
}


static int f(double t, N_Vector y, N_Vector ydot, void *user_data)
{
  int i;
  int j;
  int k;
  double Va;
  double Vs;
  double Vt;
  double v;
  double a;
  double INa, IKd, ICa,IKCa,IA,IT,Ih;
  double Ichr2;
  double Isyn;
  int Nmax=((nervous_system *)user_data)->Ntotal;
  int tipo;

  for(i=0;i<Nmax;++i) {
    j=i*NEQ;
    
    tipo=(((nervous_system *)user_data)->neuron[i].tipo);
    model_parameters *s=&(((nervous_system *)user_data)->param[tipo]);
    //printf("%d %d tipo%d\n",i,j,tipo);fflush(stdout);
 
    Va=Ith(y,1+j);
    Vs=Ith(y,2+j);
 
    if (Va<-10.0) {
      ((nervous_system *)user_data)->neuron[i].spikes.flag=1;
    }
    if ((Va>0.0)&&(((nervous_system *)user_data)->neuron[i].spikes.flag)) {
      //printf("Spike on %d at %lg top %d\n",i,t,((nervous_system *)user_data)->neuron[i].spikes.top);
      ((nervous_system *)user_data)->neuron[i].spikes.ts[((nervous_system *)user_data)->neuron[i].spikes.top]=t;
      ((nervous_system *)user_data)->neuron[i].spikes.top++;
      ((nervous_system *)user_data)->neuron[i].spikes.top=((nervous_system *)user_data)->neuron[i].spikes.top%MAXSPIKES;
      ((nervous_system *)user_data)->neuron[i].spikes.flag=0;
    }

    if (tipo==LNI) {
      Vt=s->Vt;
      /* INa current [Traub & Miles, 1991] */
      INa=(s->gNa)*Ith(y,3+j)*Ith(y,3+j)*Ith(y,4+j)*(Va-50);
      

      /* INa activation Ith(y,3+j)*/
      v=18.0+Vt-Va;
      if (ABS(v)<0.0001) {
	a=1.28-0.16*v;
      } else 
	a=0.32*v/(exp(v/4.0)-1);    
      
      Ith(ydot,3+j)=a*(1-Ith(y,3+j));
      
      v=-40.0-Vt+Va;
      if (ABS(v)<0.0001) {
	a=1.40-0.14*v;
      } else 
	a=0.28*v/(exp(v/5.0)-1.0);
      Ith(ydot,3+j)-=a*(Ith(y,3+j));
      
      /* INa inactivation Ith(y,4+j) */
      v=17+Vt-Va;
      Ith(ydot,4+j)=0.128*exp(v/18.0)*(1-Ith(y,4+j));
      v=40.0+Vt-Va;
      Ith(ydot,4+j)-=4.0*Ith(y,4+j)/(1.0+exp(v/5.0));
      
      /* IKd delayed rectifier current */
      IKd=s->gKd*Ith(y,5+j)*(Va+60);
      
      /* IKd activation Ith(y,5+j) */
      v=35.0+Vt-Va;
      if (ABS(v)<0.0001) {
	a=0.08-0.008*v;
      } else 
	a=0.016*v/(exp(v/5.0)-1.0);
      v=20.0+Vt-Va;
      Ith(ydot,5+j)=a*(1-Ith(y,5+j))-0.25*exp(v/40)*Ith(y,5+j);
      
      /* ICa current  */
      if (ABS(Vs)<0.001) {
	a=0.5*(- 24.42002442+Vs);
      } else {
	a=Vs/(1-exp(2*Vs/ 24.42002442));
      }
      ICa=s->gCa*Ith(y,6+j)*Ith(y,6+j)*Ith(y,6+j)*a;
      /* Calcium activation Ith(y,6+j) */
      Ith(ydot,6+j)=0.1*(Cagam(-Vs,39.1,2)-Ith(y,6+j));
      //printf("%lg ydot %lf\n",ICa,Ith(ydot,6+j));exit(0);
      
      /* [Ca] Ith(y,7+j)*/
      Ith(ydot,7+j)=0.001*(-0.35*ICa-s->MU*s->MU*Ith(y,7+j)+0.04*s->MU*s->MU);
      
      /* IKCa */
      IKCa=s->gKCa*Ith(y,8+j)*(Va+60.0);
      /* Activation IKCa Ith(y,8+j) */
      Ith(ydot,8+j)=3.0*Cagam(0.08,Ith(y,7+j),0.011)*(1-Ith(y,8+j))-20.0*Ith(y,8+j);
      /* Ih */
      //      Ih=gh*Ith(y,9+j)*(Vs+60.0);
      Ih=0.0;
      /* Activation Ih*/
      //Ith(ydot,9+j)=(Cagam(Vs,-80,10)-Ith(y,9+j))/(2000-1999*Cagam(Vs,-60.0,-1.0));
      Ith(ydot,9+j)=0.0;
      
      /* IA current */
      IA=s->gA*Ith(y,10+j)*(Vs+60.0);
      Ith(ydot,10+j)=(Cagam(-Vs,0,8)-Ith(y,10+j))/(350-349*Cagam(Vs,-46.0,4.0));

      Ith(ydot,11+j)=Ith(ydot,12+j)=0.0; // Padding the vector field

      Ith(ydot,13+j)=0.0;  // reserved for AMPA release
      //GABAA release
      Ith(ydot,14+j)=s->kfGABAA*(1-Ith(y,14+j))*Cagam(Va,30,-2)-s->krGABAA*Ith(y,14+j);


      // GABA-B \dot{r}=ar [T] (1-r) -br r
      Ith(ydot,15+j)= s->ar*(1.0-Ith(y,15+j))*Cagam(Va,0,-2)-s->br*Ith(y,15+j);
      // \dot{s}=K3 r - K4 s;
      Ith(ydot,16+j)=s->K3*Ith(y,15+j)-s->K4*Ith(y,16+j)  ;

      /* Axon membrane potential */
      Ith(ydot,1+j) = (1.0/s->CA)*(-s->gal*(Va+45.0 /*resting potential*/)-s->gAS*(Va-Vs)
				   -INa-IKd-IKCa);

      // Integrate synaptic input
      Isyn=synaptic_input(y,user_data,i,j);
      tmp_gaba=Ith(y,16+j);

      Ith(ydot,2+j) = (1.0/s->CS)*(-s->gsl*(Vs+45.0 /*resting potential*/)-s->gAS*(Vs-Va)
				   -ICa-IA-Isyn-((nervous_system *)user_data)->neuron[i].shift+protocolG(t,i,user_data)
				   /*+protocol(t)*/);
    
    
    LNlocal_gabaA=local_gabaA;
    LNlocal_gabaB=local_gabaB;
    
    } else if (tipo==LNE) {
     Vt=s->Vt;

      /* INa current [Traub & Miles, 1991] */
      INa=s->gNa*Ith(y,3+j)*Ith(y,3+j)*Ith(y,4+j)*(Va-50);
      
      /* INa activation Ith(y,3+j)*/
      v=18.0+Vt-Va;
      if (ABS(v)<0.0001) {
	a=1.28-0.16*v;
      } else 
	a=0.32*v/(exp(v/4.0)-1);    
      
      Ith(ydot,3+j)=a*(1-Ith(y,3+j));
      
      v=-40.0-Vt+Va;
      if (ABS(v)<0.0001) {
	a=1.40-0.14*v;
      } else 
	a=0.28*v/(exp(v/5.0)-1.0);
      Ith(ydot,3+j)-=a*(Ith(y,3+j));
      
      /* INa inactivation Ith(y,4+j) */
      v=17+Vt-Va;
      Ith(ydot,4+j)=0.128*exp(v/18.0)*(1-Ith(y,4+j));
      v=40.0+Vt-Va;
      Ith(ydot,4+j)-=4.0*Ith(y,4+j)/(1.0+exp(v/5.0));
      
      /* IKd delayed rectifier current */
      IKd=s->gKd*Ith(y,5+j)*(Va+60);
      
      /* IKd activation Ith(y,5+j) */
      v=35.0+Vt-Va;
      if (ABS(v)<0.0001) {
	a=0.08-0.008*v;
      } else 
	a=0.016*v/(exp(v/5.0)-1.0);
      v=20.0+Vt-Va;
      Ith(ydot,5+j)=a*(1-Ith(y,5+j))-0.25*exp(v/40)*Ith(y,5+j);
      
      /* ICa current  */
      if (ABS(Vs)<0.001) {
	a=0.5*(- 24.42002442+Vs);
      } else {
	a=Vs/(1-exp(2*Vs/ 24.42002442));
      }
      ICa=s->gCa*Ith(y,6+j)*Ith(y,6+j)*Ith(y,6+j)*a;
      /* Calcium activation Ith(y,6+j) */
      Ith(ydot,6+j)=0.1*(Cagam(-Vs,39.1,2)-Ith(y,6+j));
      //printf("%lg ydot %lf\n",ICa,Ith(ydot,6+j));exit(0);
      
      /* [Ca] Ith(y,7+j)*/
      Ith(ydot,7+j)=0.001*(-0.35*ICa-s->MU*s->MU*Ith(y,7+j)+0.04*s->MU*s->MU);
      
      /* IKCa */
      IKCa=s->gKCa*Ith(y,8+j)*(Va+60.0);
      /* Activation IKCa Ith(y,8+j) */
      Ith(ydot,8+j)=3.0*Cagam(0.09,Ith(y,7+j),0.011)*(1-Ith(y,8+j))-20.0*Ith(y,8+j);
      
      /* Ih */
      //Ih=s->gh*Ith(y,9+j)*(Vs+60.0);
      Ih=0.0;
      /* Activation Ih*/
      //Ith(ydot,9+j)=(Cagam(Vs,-80,10)-Ith(y,9+j))/(2000-1999*Cagam(Vs,-60.0,-1.0));
      Ith(ydot,9+j)=0.0;
      
      /* IA */
      IA=s->gA*Ith(y,10+j)*(Vs+60.0);
      Ith(ydot,10+j)=(Cagam(-Vs,0,8)-Ith(y,10+j))/(350-349*Cagam(Vs,-46.0,4.0));
      
      
      Ith(ydot,11+j)=0.0;      
      Ith(ydot,12+j)=0.0;
      /* AMPA neurotransmitter release */
      Ith(ydot,13+j)= s->kfAMPA*(1-Ith(y,13+j))*Cagam(Va,30,-2)-s->krAMPA*Ith(y,13+j);
      Ith(ydot,14+j)=0.0; // Reserved for GABA release if needed

      // PN release
      //   Ith(ydot,15+j)= s->kfPN*(1-Ith(y,15+j))*Cagam(Vs,30,-2)-s->krPN*Ith(y,15+j);
      Ith(ydot,15+j)=0.0;

      //Ichr2=(gch2)*Ith(y,16+j)*Vs;

      //Ith(ydot,16+j)=0.4*protocol2(t)*(1-Ith(y,16+j))-0.0001/*0.00005*/*Ith(y,16+j);
      Ith(ydot,16+j)=0.0;


      /* Axon membrane potential */
      Ith(ydot,1+j) = (1.0/s->CA)*(-s->gal*(Va+45.0 /*resting potential*/)-s->gAS*(Va-Vs)
				   -INa-IKd-IKCa);
      // Integrate synaptic input
      Isyn=synaptic_input(y,user_data,i,j);
      //if (fabs(Isyn)>0.001) printf("%lg %d %d\n",Isyn,i,j);

      /* Soma membrane pontential */
      Ith(ydot,2+j) = (1.0/s->CS)*(-s->gsl*(Vs+45.0 /*resting potential*/)-s->gAS*(Vs-Va)
				   -ICa/*-Ih*/-IA-((nervous_system *)user_data)->neuron[i].shift-Isyn
				   /*-Ichr2*//*+protocol(t)*//*+protocolG(t,i,user_data)*/);

  
    } else if (tipo==PN) {
      Vt=s->Vt;

      /* INa current [Traub & Miles, 1991] */
      INa=s->gNa*Ith(y,3+j)*Ith(y,3+j)*Ith(y,4+j)*(Va-50);
      
      /* INa activation Ith(y,3+j)*/
      v=18.0+Vt-Va;
      if (ABS(v)<0.0001) {
	a=1.28-0.16*v;
      } else 
	a=0.32*v/(exp(v/4.0)-1);    
      
      Ith(ydot,3+j)=a*(1-Ith(y,3+j));
      
      v=-40.0-Vt+Va;
      if (ABS(v)<0.0001) {
	a=1.40-0.14*v;
      } else 
	a=0.28*v/(exp(v/5.0)-1.0);
      Ith(ydot,3+j)-=a*(Ith(y,3+j));
      
      /* INa inactivation Ith(y,4+j) */
      v=17+Vt-Va;
      Ith(ydot,4+j)=0.128*exp(v/18.0)*(1-Ith(y,4+j));
      v=40.0+Vt-Va;
      Ith(ydot,4+j)-=4.0*Ith(y,4+j)/(1.0+exp(v/5.0));
      
      /* IKd delayed rectifier current */
      IKd=s->gKd*Ith(y,5+j)*(Va+60);
      
      /* IKd activation Ith(y,5+j) */
      v=35.0+Vt-Va;
      if (ABS(v)<0.0001) {
	a=0.08-0.008*v;
      } else 
	a=0.016*v/(exp(v/5.0)-1.0);
      v=20.0+Vt-Va;
      Ith(ydot,5+j)=a*(1-Ith(y,5+j))-0.25*exp(v/40)*Ith(y,5+j);
      
      /* ICa current  */
      if (ABS(Vs)<0.001) {
	a=0.5*(- 24.42002442+Vs);
      } else {
	a=Vs/(1-exp(2*Vs/ 24.42002442));
      }
      ICa=s->gCa*Ith(y,6+j)*Ith(y,6+j)*Ith(y,6+j)*a;
      /* Calcium activation Ith(y,6+j) */
      Ith(ydot,6+j)=0.1*(Cagam(-Vs,39.1,2)-Ith(y,6+j));
      //printf("%lg ydot %lf\n",ICa,Ith(ydot,6+j));exit(0);
      
      /* [Ca] Ith(y,7+j)*/
      Ith(ydot,7+j)=0.001*(-0.35*ICa-s->MU*s->MU*Ith(y,7+j)+0.04*s->MU*s->MU);
      
      /* IKCa */
      IKCa=s->gKCa*Ith(y,8+j)*(Va+60.0);
      /* Activation IKCa Ith(y,8+j) */
      Ith(ydot,8+j)=3.0*Cagam(0.09,Ith(y,7+j),0.011)*(1-Ith(y,8+j))-20.0*Ith(y,8+j);
      
      /* Ih */
      //Ih=s->gh*Ith(y,9+j)*(Vs+60.0);
      Ih=0.0;
      /* Activation Ih*/
      //Ith(ydot,9+j)=(Cagam(Vs,-80,10)-Ith(y,9+j))/(2000-1999*Cagam(Vs,-60.0,-1.0));
      Ith(ydot,9+j)=0.0;
      
      /* IA */
      IA=s->gA*Ith(y,10+j)*(Vs+60.0);
      Ith(ydot,10+j)=(Cagam(-Vs,0,8)-Ith(y,10+j))/(350-349*Cagam(Vs,-46.0,4.0));
      
      
      Ith(ydot,11+j)=0.0;      
      Ith(ydot,12+j)=0.0;
      /* AMPA neurotransmitter release */
      Ith(ydot,13+j)= s->kfAMPA*(1-Ith(y,13+j))*Cagam(Va,30,-2)-s->krAMPA*Ith(y,13+j);
      Ith(ydot,14+j)=0.0; // Reserved for GABA release if needed

      // PN release
      //   Ith(ydot,15+j)= s->kfPN*(1-Ith(y,15+j))*Cagam(Vs,30,-2)-s->krPN*Ith(y,15+j);
      Ith(ydot,15+j)=0.0;

      //Ichr2=(gch2)*Ith(y,16+j)*Vs;

      //Ith(ydot,16+j)=0.4*protocol2(t)*(1-Ith(y,16+j))-0.0001/*0.00005*/*Ith(y,16+j);
      Ith(ydot,16+j)=0.0;


      /* Axon membrane potential */
      Ith(ydot,1+j) = (1.0/s->CA)*(-s->gal*(Va+45.0 /*resting potential*/)-s->gAS*(Va-Vs)
				   -INa-IKd-IKCa);
      // Integrate synaptic input
      Isyn=synaptic_input(y,user_data,i,j);
      //if (fabs(Isyn)>0.001) printf("%lg %d %d\n",Isyn,i,j);

      /* Soma membrane pontential */
      Ith(ydot,2+j) = (1.0/s->CS)*(-s->gsl*(Vs+45.0 /*resting potential*/)-s->gAS*(Vs-Va)
				   -ICa/*-Ih*/-IA-((nervous_system *)user_data)->neuron[i].shift-Isyn
				   /*-Ichr2*//*+protocol(t)*/+protocolG(t,i,user_data));


    PNlocal_gabaB=local_gabaB;
    PNlocal_gabaA=local_gabaA;
    }
  }
  return(0);      
}


/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */
static void PrintAverage(FILE *br,double t, N_Vector y,nervous_system *NS,double PNMA,double PNMB,double LNMA,double LNMB) {
  int i=0;
  double sum=0;
  fprintf(br,"%lg ",t/1000.0);
  for(i=0;i<NS->Nlocal[0];++i)
    sum+=Ith(y,2+i*NEQ);
  sum/=(double) NS->Nlocal[0];
  fprintf(br,"%lg ",sum);
  sum=0;
  for(;i<(NS->Nlocal[0]+NS->Nlocal[1]);++i)
    sum+=Ith(y,2+i*NEQ);
  sum/=(double) NS->Nlocal[1];
  fprintf(br,"%lg ",sum);
  fprintf(br,"%lg %lg ",PNMA,PNMB);
  fprintf(br,"%lg %lg ",LNMA,LNMB);
  fprintf(br,"%lg %lg\n",peak_gabaA,peak_gabaB);

  /*  sum=0;
  for(;i<(NS->Nlocal[0]+NS->Nlocal[1]+NS->Nlocal[2]);++i)
    sum+=Ith(y,2+i*NEQ);
  sum/=(double) NS->Nlocal[2];
  fprintf(br,"%lg\n",sum);*/

}

void save_final_conditions(N_Vector y,nervous_system *NS) {
    FILE *ar;
    int i=0;
    int j;

    ar=fopen("final_conditions.dat","w");

    if (ar!=NULL) {

	for(j=0;j<NEQ;++j)
	    fprintf(ar,"%lg\n",Ith(y,j+i*NEQ+1));
	i=NS->Nlocal[0];
	for(j=0;j<NEQ;++j)
	    fprintf(ar,"%lg\n",Ith(y,j+i*NEQ+1));
	i=NS->Nlocal[0]+NS->Nlocal[1];
	for(j=0;j<NEQ;++j)
	    fprintf(ar,"%lg\n",Ith(y,j+i*NEQ+1));

	fclose(ar);
    }
}
		    


static void PrintOutput(FILE *ar, double t, N_Vector y,nervous_system *NS)
{
    int i;
 fprintf(ar,"%lg",t/1000.0);
 for(i=0;i<NS->Nlocal[0]+NS->Nlocal[1];++i) 
   fprintf(ar," %12.4lg",Ith(y,2+i*NEQ));
 
 // for(i=NS->Nlocal[0];i<NS->Nlocal[0]+NS->Nlocal[1];++i) 
 //  fprintf(ar," %12.4lg",Ith(y,7+i*NEQ));

 fprintf(ar,"\n");

  return;
}

static void PrintOutputCa(FILE *dr, double t, N_Vector y,nervous_system *NS)
{
    int i;
 fprintf(dr,"%lg",t/1000.0);
 double sum=0;
 //for(i=0;i<NS->Nlocal[0]+NS->Nlocal[1];++i) ^M
 //  fprintf(dr," %12.4lg",Ith(y,7+i*NEQ));^M
 for(i=0;i<NS->Nlocal[0];++i)
   sum+=Ith(y,7+i*NEQ);
 sum/=(double) NS->Nlocal[0];
 fprintf(dr," %12.4lg\n",sum);
 fflush(dr);
 // for(i=NS->Nlocal[0];i<NS->Nlocal[0]+NS->Nlocal[1];++i) ^M
 //  fprintf(ar," %12.4lg",Ith(y,2+i*NEQ));^M

         //fprintf(dr,"\n");^M

  return;

}

 void set_initial_conditions(N_Vector y,int Nmax,nervous_system *NS) {
  int i;
  int j;
  int k;
  FILE *ar=NULL;


  if (Nmax<=0) {
    puts("No neurons in the simulation");
    exit(0);
  }
  
  //ar=fopen("final_conditions.dat","r");

  if (ar!=NULL) {
      puts("READING from file initial conditions");
      double hcrt[NEQ],gaba[NEQ],lc[NEQ];
      for(i=0;i<NEQ;++i)
	  fscanf(ar,"%lg",&hcrt[i]);
      
      for(i=0;i<NEQ;++i)
	  fscanf(ar,"%lg",&gaba[i]);

      for(i=0;i<NEQ;++i)
	  fscanf(ar,"%lg",&lc[i]);
      

      for(i=0;i<NS->Nlocal[0];++i)
	  for(j=0;j<NEQ;++j)
	      Ith(y,j+i*NEQ+1)=hcrt[j]+0.1*genrand();

      
      for(;i<NS->Nlocal[0]+NS->Nlocal[1];++i)
	  for(j=0;j<NEQ;++j)
	      Ith(y,j+i*NEQ+1)=gaba[j]+0.1*genrand();

       for(;i<NS->Nlocal[0]+NS->Nlocal[1]+NS->Nlocal[2];++i)
	  for(j=0;j<NEQ;++j)
	      Ith(y,j+i*NEQ+1)=lc[j]+0.1*genrand();

       for(i=0;i<NS->Nlocal[0]+NS->Nlocal[1]+NS->Nlocal[2];++i)
	   Ith(y,14+NEQ*i)=Ith(y,13+NEQ*i)=Ith(y,15+NEQ*i)=Ith(y,16+NEQ*i)=0.0;
       fclose(ar);
  } else {
  for(k=0;k<Nmax;++k) {
      j=NEQ*k;
      //for(i=3;i<=NEQ;++i)   Ith(y,i+j)=0.2*genrand();
      for(i=3;i<=NEQ;++i)   Ith(y,i+j)=0.0;
      
      //Ith(y,1+j) = -61+10.0*genrand();
      //Ith(y,2+j) = -61+10.0*genrand();
      //Ith(y,7+j) = 0.04+0.01*genrand();
      Ith(y,1+j) =-60.0;
      Ith(y,2+j) =-60.0;
      Ith(y,7+j) = 0.04;
   }
  
  }
 }

/* 
 * Get and print some final statistics
 */

static void PrintFinalStats(void *cvode_mem)
{
  long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
  int flag;

  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  flag = CVDlsGetNumJacEvals(cvode_mem, &nje);
  check_flag(&flag, "CVDlsGetNumJacEvals", 1);
  flag = CVDlsGetNumRhsEvals(cvode_mem, &nfeLS);
  check_flag(&flag, "CVDlsGetNumRhsEvals", 1);

  flag = CVodeGetNumGEvals(cvode_mem, &nge);
  check_flag(&flag, "CVodeGetNumGEvals", 1);

  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
	 nst, nfe, nsetups, nfeLS, nje);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
	 nni, ncfn, netf, nge);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer 
 */

static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
	      funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  return(0);
}
