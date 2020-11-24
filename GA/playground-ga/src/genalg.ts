/* A simple Genetic Algorithm and the interaction with the NN via the fitness function */
/* The program is obviously adapted from the earlier PSO program */
const P_R = 0.5;
const P_M = 0.5;
const POPSZ = 100; 	// population size
const T	= 10;		// learning steps (SGD) in for fitness evaluation

import * as nn from "./nn";
import {
  State,
  datasets,
  getKeyFromValue,
} from "./state";
import {Example2D} from "./dataset";

let state = State.deserializeState();

export class Individual {	// actually a class for individuals is not really needed for GA
	x: number[] = [];	/* this will be part of the genetic code (link on or off) */
	y: number[] = [];	/* this will be another part of the genetic code (initialisations) */
	f: number;		/* fitness */
	d: number; 		/* (max) length of individual (size of search space) */
	constructor(dim: number) { 	/* when generated the length must be known */
		this.d = dim;
		for (let i = 0; i < this.d; i++) {
			if (Math.random() < 0.5) this.x[i] = 0;
			else this.x[i] = 1;
			if (Math.random() < 0.5) this.y[i] = 0;
			else this.y[i] = 1;
			this.f = Number.MAX_VALUE;
		}
	}
	mutate() {
		for (let j = 0; j < this.d; j++) {
			if (Math.random() < P_M)  {
				this.x[j] = 1 - this.x[j];
			}
			if (Math.random() < P_M)  {
				this.y[j] = 1 - this.y[j];
			}
		}
	}
}

export class Population {
	individuals: Individual[] = [];
	indiv: Individual;
	bestindiv: Individual; 	/* best individual */
	fb: number; 		/* best fitness */
	ib: number;		/* index of best */
	fm: number;	  	/* mean fitness: selection is simple: here an individual */
				/* gets selected when better than mean, this can be */
				/* improved of course, but may not be critical here. */
	dim: number;		/* string length */

	updateBestFitness(f: number, i: number){  /* It is actually a minimisation problem. */
		/* In contrast to PSO, this is not used by GA, but is good to know. */
		if (f < this.fb) {
			this.fb = f;
			for (let j = 0; j < this.dim; j++) {
				this.bestindiv.x[j] = this.individuals[i].x[j];
				this.bestindiv.y[j] = this.individuals[i].y[j];
			}
		}
	}

	diversity(): number { // a simple component-wise measure of diversity in the population
		let dv = 0;
		for (let j = 0; j < this.dim; j++) {
			let z=0;
			for (let i = 0; i < POPSZ; i++) {
				z += 2 * (this.individuals[i].x[j] - 0.5); // should be near zero if diverse
				z += 2 * (this.individuals[i].y[j] - 0.5); // should be near zero if diverse
			}
			z /= POPSZ;
			dv += Math.abs(z);
		}
		return(1 - dv / (2*this.dim)); // 1 means divers and 0 collapse (but some weights are unused)
	}	
			
	crossOver(p1: number, p2: number): Individual { //This is insensitive to network structure
		let descendant = new Individual(this.dim);
		let cut = Math.floor(Math.random() * (this.dim-1)+1);
		for (let j = 0; j < cut; j++) {
			descendant.x[j]=this.individuals[p1].x[j];
			descendant.y[j]=this.individuals[p1].y[j];
		}
		for (let j = cut; j < this.dim; j++) {
			descendant.x[j]=this.individuals[p2].x[j];
			descendant.y[j]=this.individuals[p2].y[j];
		}
		descendant.f = Number.MAX_VALUE;
		return descendant;	
	}

	updatePopulation(network: nn.Node[][], trainData: Example2D[], testData: Example2D[]): number {

		this.fb = 1000.0; 	/* initialisation for minimisation problems */
		this.fm = 0; 		/* no selection before first fitness update */
		this.ib = 1;		/* index of best (not yet known) */

		let intpop = new Array(POPSZ); /* indices of intermediate population */
		let unsel = new Array(POPSZ);	/* indices of individuals to be unselected */
		for (let i = 0; i < POPSZ; i++) { // uses both data (sub)set in different ways
			this.individuals[i].f = getFitness(network,trainData,testData,this.individuals[i].x,this.individuals[i].y,this.dim);
			/* this is as much as the GA knows about the problem */
			/* with different data sets it can get different fitnesses */
			/* (for MOO this may be intended, but is not needed here). *.
			/* The network pointer is passed back and forth for technical  */
			/* reasons, i.e. the playground asks the GA:  */ 
			/* Update my network! and then best network will be shown. */

			if (this.individuals[i].f < this.fb) {	/* best fitness */
				this.fb = this.individuals[i].f;
				this.ib = i;		
			}
			this.fm += this.individuals[i].f; /* calculate mean fitnesses */
		}
		this.updateBestFitness(this.fb,this.ib);
		this.fm = this.fm / POPSZ;

		let k = 0;
		let l = 0;
		for (let i = 0; i < POPSZ; i++) {
			if (this.individuals[i].f <= this.fm ) {  	// minimisation ! 
				intpop[k]=i; 	// indexing intermediate population 
				k++;
			}
			else {
				unsel[l]=i;	// to be overwritten in place 
				l++;
			}
		}
		let kk = k;
		let ll = l;
		for (let i = 0; i < ll; i++) {
			let p1 = Math.floor(Math.random() * kk);
			if (Math.random() > P_R) {  // just copy, no crossover 
				for (let j = 0; j < this.dim; j++) {
					this.individuals[unsel[i]].x[j]=this.individuals[intpop[p1]].x[j];
					this.individuals[unsel[i]].y[j]=this.individuals[intpop[p1]].y[j];
				}
			}
			else {				// otherwise do crossover
				let p2 = Math.floor(Math.random() * kk); // could be same ? 
				this.individuals[unsel[i]]=this.crossOver(intpop[p1],intpop[p2]);
			}
		}
		for (let i = 0; i < POPSZ; i++) {
			this.individuals[i].mutate();
		}

		// finally the the network to be displayed to the current best
		
		this.fb = getFitness(network,trainData,testData,this.individuals[this.ib].x,this.individuals[this.ib].y,this.dim);
		//return(this.fb);
		return(this.diversity());
	}
}

export function	buildPopulation(nnDim: number): Population {
	let popul = new Population;
	popul.dim = nnDim;		/* (max) length of string */
	for (let i = 0; i < POPSZ; i++) {
		let indiv = new Individual(popul.dim); 
		popul.individuals.push(indiv);
	}
	popul.fb = Number.MAX_VALUE;	/* for minimisation problem */
	for (let j = 0; j < this.d; j++) {
		this.bestindiv.j[j] = this.individuals[0].x[j];
		this.bestindiv.y[j] = this.individuals[0].y[j];
	}
	return popul;
}

/* In this function neural network feedforwards each data point once */

export function getFitness(network: nn.Node[][], trainData: Example2D[], testData: Example2D[], x: number[], y: number[], dim: number): number {
	//let nnn = nn.setWeights(network, x, dim); //this was for PSO: assign x to weights, this is 
						// just for comparison, x is now a binary
						// vector, so this wouldn't really work
						// unless something like "weightless" 
						// networks are considered, which is, 
						// however, not relevant here 
	nn.setLinkWeights(network, x, y, dim); /* GA: set or delete links */
				/* At this point may like to addd a check whether it went well, for the 
				PSO we just checked whether the dimension was consistent. */
	for (let t=0; t<T; t++) { 	/* perform a number of learning steps according to parameter T*/
		oneStepForGaFitness(network,trainData); // in playground.ts a similar functions is used for 
						// the default network, but here for each individual 
	}
	return(getLoss(network, testData)); /* Different data set from network training in oneStep above */
	//return(Math.random());
}

/* the following ones are essentially copies from playground.ts */

function oneStepForGaFitness(netwrk: nn.Node[][], trainData: Example2D[]): void {
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(netwrk, input);
    nn.backProp(netwrk, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(netwrk, state.learningRate, 0);
    }
  });
  // Compute the loss.
  // let lossTrain = getLoss(netwrk, trainData);
  // let lossTest = getLoss(netwrk, testData);	// We check these only when GA takes over again.
  // updateUI();
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}
