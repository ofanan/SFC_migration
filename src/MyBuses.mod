int nbKids=300;

// a tuple is like a struct in C, a class in C++ or a record in Pascal
tuple bus // Each bus has its own num of seats, and cost
{
key int id;
float dummy;
int nbSeats;
float cost;
}

// This is a tuple set
{bus} buses = ...; // This will hold the sol?

// decision variable array
dvar int+ nbBus[buses]; // nbBus[i] will hold the # of buses 

// objective
minimize
 sum(b in buses) b.cost*nbBus[b];

// constraints
subject to
{
 sum(b in buses) b.nbSeats * nbBus[b] >= nbKids;
 sum(b in buses) b.dummy   * nbBus[b] <= 100000; 
}

tuple solution
{
  
  int nbBus;
  int sizeBus;
}

{solution} solutions={<nbBus[b],b.id> | b in buses};
