import type {
	ActivationData,
	GraphData,
	GraphNode,
	GraphLink,
	SpecialistNeuron,
	SignalType
} from '$lib/types';
import { getSignalType } from './colorScales';

/**
 * Build a specialist neuron lookup map for quick access
 */
function buildSpecialistMap(specialists: SpecialistNeuron[]): Map<string, SpecialistNeuron> {
	const map = new Map<string, SpecialistNeuron>();
	for (const spec of specialists) {
		const key = `L${spec.layer}_N${spec.neuron_idx}`;
		map.set(key, spec);
	}
	return map;
}

/**
 * Build static graph structure (nodes and links) without token-specific activations.
 * This is called once when data/sequence/layers change, NOT on every token change.
 */
export function buildStaticGraphData(
	data: ActivationData,
	sequenceIdx: number = 0,
	expandedLayers: boolean[] = new Array(8).fill(true)
): GraphData {
	const sequence = data.sequences[sequenceIdx];
	if (!sequence) {
		return { nodes: [], links: [] };
	}

	const specialistMap = buildSpecialistMap(data.specialists);
	const nodes: GraphNode[] = [];
	const links: GraphLink[] = [];

	const LAYER_SPACING = 100;

	// Build nodes for each layer (without token-specific activations)
	for (const layerAct of sequence.layer_activations) {
		const { layer_idx, sampled_indices } = layerAct;

		if (!expandedLayers[layer_idx]) continue;

		sampled_indices.forEach((neuronIdx) => {
			const nodeId = `L${layer_idx}_N${neuronIdx}`;
			const specialist = specialistMap.get(nodeId);

			nodes.push({
				id: nodeId,
				layer: layer_idx,
				neuron_idx: neuronIdx,
				is_specialist: !!specialist,
				specialist_char: specialist?.trigger_char,
				excitatory_activation: 0, // Will be updated per token
				inhibitory_activation: 0,
				signal_type: 'inactive',
				z: layer_idx * LAYER_SPACING
			});
		});
	}

	// Build inter-layer edges (connect all adjacent layer nodes, weight updated per token)
	const nodesByLayer = new Map<number, GraphNode[]>();
	for (const node of nodes) {
		const layerNodes = nodesByLayer.get(node.layer) || [];
		layerNodes.push(node);
		nodesByLayer.set(node.layer, layerNodes);
	}

	// Create sparse connections between adjacent layers
	for (let l = 0; l < data.metadata.n_layers - 1; l++) {
		if (!expandedLayers[l] || !expandedLayers[l + 1]) continue;

		const currentLayerNodes = nodesByLayer.get(l) || [];
		const nextLayerNodes = nodesByLayer.get(l + 1) || [];

		// Connect a subset of nodes (limit for performance)
		const maxConnections = 10;
		const step1 = Math.max(1, Math.floor(currentLayerNodes.length / maxConnections));
		const step2 = Math.max(1, Math.floor(nextLayerNodes.length / maxConnections));

		for (let i = 0; i < currentLayerNodes.length; i += step1) {
			for (let j = 0; j < nextLayerNodes.length; j += step2) {
				links.push({
					source: currentLayerNodes[i].id,
					target: nextLayerNodes[j].id,
					weight: 0.1, // Will be updated per token
					type: 'inter_layer'
				});
			}
		}
	}

	return { nodes, links };
}

/**
 * Convert activation data at a specific token position to graph nodes and links
 * @deprecated Use buildStaticGraphData + getTokenActivations for better performance
 */
export function buildGraphData(
	data: ActivationData,
	tokenIdx: number,
	sequenceIdx: number = 0,
	expandedLayers: boolean[] = new Array(8).fill(true)
): GraphData {
	const sequence = data.sequences[sequenceIdx];
	if (!sequence) {
		return { nodes: [], links: [] };
	}

	const specialistMap = buildSpecialistMap(data.specialists);
	const nodes: GraphNode[] = [];
	const links: GraphLink[] = [];

	// Layer spacing along Z-axis
	const LAYER_SPACING = 100;

	// Build nodes for each layer
	for (const layerAct of sequence.layer_activations) {
		const { layer_idx, sampled_indices, excitatory, inhibitory } = layerAct;

		// Skip collapsed layers
		if (!expandedLayers[layer_idx]) continue;

		sampled_indices.forEach((neuronIdx, i) => {
			const nodeId = `L${layer_idx}_N${neuronIdx}`;
			const specialist = specialistMap.get(nodeId);

			// Get activations for current token
			const excAct = excitatory[tokenIdx]?.[i] ?? 0;
			const inhAct = inhibitory[tokenIdx]?.[i] ?? 0;

			const signalType: SignalType = getSignalType(excAct, inhAct);

			nodes.push({
				id: nodeId,
				layer: layer_idx,
				neuron_idx: neuronIdx,
				is_specialist: !!specialist,
				specialist_char: specialist?.trigger_char,
				excitatory_activation: excAct,
				inhibitory_activation: inhAct,
				signal_type: signalType,
				// Initial position hint based on layer
				z: layer_idx * LAYER_SPACING
			});
		});
	}

	// Build inter-layer edges (sequential flow between active neurons)
	const nodesByLayer = new Map<number, GraphNode[]>();
	for (const node of nodes) {
		const layerNodes = nodesByLayer.get(node.layer) || [];
		layerNodes.push(node);
		nodesByLayer.set(node.layer, layerNodes);
	}

	// Connect active neurons between adjacent layers
	for (let l = 0; l < data.metadata.n_layers - 1; l++) {
		if (!expandedLayers[l] || !expandedLayers[l + 1]) continue;

		const currentLayerNodes = nodesByLayer.get(l) || [];
		const nextLayerNodes = nodesByLayer.get(l + 1) || [];

		// Filter to active neurons only
		const activeCurrents = currentLayerNodes
			.filter((n) => n.excitatory_activation > 0.1 || n.inhibitory_activation > 0.1)
			.slice(0, 15); // Limit connections for performance

		const activeNexts = nextLayerNodes
			.filter((n) => n.excitatory_activation > 0.1 || n.inhibitory_activation > 0.1)
			.slice(0, 15);

		for (const source of activeCurrents) {
			for (const target of activeNexts) {
				const weight =
					Math.min(
						source.excitatory_activation + source.inhibitory_activation,
						target.excitatory_activation + target.inhibitory_activation
					) / 2;

				if (weight > 0.05) {
					links.push({
						source: source.id,
						target: target.id,
						weight: Math.min(weight, 1),
						type: 'inter_layer'
					});
				}
			}
		}
	}

	return { nodes, links };
}

/**
 * Get activation values for a specific token across all nodes
 */
export function getTokenActivations(
	data: ActivationData,
	tokenIdx: number,
	sequenceIdx: number = 0
): Map<string, { excitatory: number; inhibitory: number }> {
	const activations = new Map<string, { excitatory: number; inhibitory: number }>();
	const sequence = data.sequences[sequenceIdx];

	if (!sequence) return activations;

	for (const layerAct of sequence.layer_activations) {
		const { layer_idx, sampled_indices, excitatory, inhibitory } = layerAct;

		sampled_indices.forEach((neuronIdx, i) => {
			const nodeId = `L${layer_idx}_N${neuronIdx}`;
			activations.set(nodeId, {
				excitatory: excitatory[tokenIdx]?.[i] ?? 0,
				inhibitory: inhibitory[tokenIdx]?.[i] ?? 0
			});
		});
	}

	return activations;
}

/**
 * Calculate layer statistics for a given token
 */
export function getLayerStats(
	data: ActivationData,
	tokenIdx: number,
	sequenceIdx: number = 0
): {
	layer: number;
	activeCount: number;
	excitatoryCount: number;
	inhibitoryCount: number;
	sparsity: number;
}[] {
	const sequence = data.sequences[sequenceIdx];
	if (!sequence) return [];

	return sequence.layer_activations.map((layerAct) => {
		const { layer_idx, sampled_indices, excitatory, inhibitory, sparsity } = layerAct;

		let activeCount = 0;
		let excitatoryCount = 0;
		let inhibitoryCount = 0;

		sampled_indices.forEach((_, i) => {
			const excAct = excitatory[tokenIdx]?.[i] ?? 0;
			const inhAct = inhibitory[tokenIdx]?.[i] ?? 0;

			if (excAct > 0.05 || inhAct > 0.05) {
				activeCount++;
				if (excAct >= inhAct) {
					excitatoryCount++;
				} else {
					inhibitoryCount++;
				}
			}
		});

		return {
			layer: layer_idx,
			activeCount,
			excitatoryCount,
			inhibitoryCount,
			sparsity
		};
	});
}
