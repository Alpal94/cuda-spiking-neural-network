<?php

$spiking = new SpikingNetwork;
$spiking->generateNeurons(25);
/*$spiking->freshNode(0, array(5, 1), array(2, 1));
$spiking->freshNode(1, array(2, 3), array(1,1));
$spiking->freshNode(2, array(1, 3), array(1,1));
$spiking->freshNode(3, array(1, 2, 4, 5), array(1,1,1,1));
$spiking->freshNode(4, array(1, 2), array(1,1));
$spiking->freshNode(5, array(1, 2), array(1,1));*/

/*
 * INPUT Neurons:
 * 0,1,2 ... 7,8 --> retina vision sensors
 * 9 --> pleasure touch sensor
 * 10 --> pain touch sensor
 * 11 --> signal generator
 *
 * OUTPUT Neurons:
 * 12 --> forward motor neuron (output)
 * 13 --> right motor neuron (output)
 * 14 --> left motor neuron (output)
 *
 */

//Retina to pleasure
retina(0,0,0,15, $spiking);
retina(0,0,1,16, $spiking);
retina(0,1,0,17, $spiking);
retina(0,1,1,18, $spiking);
//Retina to pain
retina(0,0,0,19, $spiking);
retina(0,0,1,20, $spiking);
retina(0,1,0,21, $spiking);
retina(0,1,1,22, $spiking);
//Pleasure to single node
deepRetina(15,15,15,23, $spiking);
//Pain to single node
deepRetina(19,19,19,24, $spiking);

connect(23, 9, $spiking); //To pleasure touch sensor
connect(24, 10, $spiking); //To pain touch sensor

connect(23, 12, $spiking); //To forward movement motor neuron
connect(24, 13, $spiking); //To right movement motor neuron

connect(14, 12, $spiking); //Connect left to forward motor neuron
connect(14, 13, $spiking); //Connect left to right motor neuron
connect(14, 11, $spiking); //Left motor neuron connect to signal generator



/*$spiking->layerNeurons(1, 5);
$spiking->layerNeurons(5, 8);
$spiking->connectLayers(1, 3, 5, 4);*/
//$spiking->multipleLayerNeurons(0);
//$spiking->connectLayers( 3 * 40 , 4 * 40 );

function retina($f, $si, $sj, $e, $spiking) {
	$count = array(); $num = $f;
	for($i = $f; $i < 3 + $f; $i++)
		for($j = $f; $j < 3 + $f; $j++) {
			$count[$i][$j] = $num;
			$num++;
		}

	for($i = $si; $i < 2 + $si; $i++) {
		for($j = $sj; $j < 2 + $sj; $j++) {
			$spiking->freshNode($count[$i][$j], array($e), "");
		}
	}
}
function deepRetina($f, $si, $sj, $e, $spiking) {
	$count = $f;
	for($i = $si; $i < 2 + $si; $i++) {
		for($j = $sj; $j < 2 + $sj; $j++) {
			$spiking->freshNode($count, array($e), "");
			$count++;
		}
	}
}

function connect($from, $to, $spiking) {
	$spiking->freshNode($from, array($to), "");
}
$spiking->printNetwork();

class SpikingNetwork {
	private $noNeurons = 1000;
	private $nodes = array();
	private $synapse_weight = 10;

	function generateNeurons($noNeurons) {
		$this->noNeurons = $noNeurons;
		for($i = 0; $i < $this->noNeurons; $i++) {
			$node = new NODE;
			$node->init($i);
			array_push($this->nodes, $node);
		}
	}

	/*
	 * Create a fully connected layer of neurons. 
	 */
	function layerNeurons($start, $layer) {
		$width = $layer - $start;
		if(!$layer) $layer = $width;

		//Connect each layer to next layer
		for($i = $start; $i < $width + $start; $i++) for($j = 0; $j < $width; $j++) {
			$this->nodes[$i]->addNode($width + $start + $j, $this->synapse_weight);
			$this->nodes[$width + $start + $j]->addNode($i, $this->synapse_weight);
		}
	}

	function freshNode($id, $neighbors, $weights) {
		for($i = 0; $i < count($neighbors); $i++) {
			$weight = 1;
			if(isset($weights[$i])) $weight = $weights[$i];
			$this->nodes[$id]->addNode($neighbors[$i], $weight);
			$this->nodes[$neighbors[$i]]->addNode($id, $weight);
		}
	}

	/*
	 * Check for duplicates.  Return true if duplicate found, false otherwise.
	 */
	private function checkDuplicates($currNode, $neighbor) {
		for($j = 0; $j < count($this->nodes[$currNode]->neighbors); $j++)
			if($this->nodes[$currNode]->neighbors[$j] == $neighbor) return true;
		return false;
	}

	/*
	 * Create multiple layers of neurons.
	 */
	function multipleLayerNeurons($start) {
		for($i = 0; $i < 10; $i++) 
			$this->layerNeurons( $start + $i , 0 );
	}

	function connectLayers($layerI, $widthI, $layerJ, $widthJ) {
		//Connect each layer to next layer
		for($i = $layerI; $i < $layerI + $widthI; $i++) for($j = 0; $j < $widthJ; $j++) {
			if(!$this->checkDuplicates($i, $layerJ + $j)) 
				$this->nodes[$i]->addNode($layerJ + $j, $this->synapse_weight);
			if(!$this->checkDuplicates($layerJ + $j, $i)) 
				$this->nodes[$layerJ + $j]->addNode($i, $this->synapse_weight);
		}
	}

	function printNetwork() {
		echo "$this->noNeurons\n";
		for($i = 0; $i < $this->noNeurons; $i++) {
			echo count($this->nodes[$i]->neighbors);
			echo " ";
			for($j = 0; $j < count($this->nodes[$i]->neighbors); $j++) {
				echo $this->nodes[$i]->neighbors[$j];
				echo " ";
			}
			for($j = 0; $j < count($this->nodes[$i]->synapses); $j++) {
				echo $this->nodes[$i]->synapses[$j];
				echo " ";
			}
			echo "\n";
		}
	}
}

class NODE {
	public $id;
	public $neighbors = array();
	public $synapses = array();

	function init($id) {
		$this->id = $id;
	}

	function addNode($node, $weight) {
		array_push($this->neighbors, $node);
		array_push($this->synapses , $weight);
	}
}

?>
