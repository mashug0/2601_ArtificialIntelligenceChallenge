import type { ActivationData } from '$lib/types';

/**
 * Load activation data from the static folder
 */
export async function loadActivationData(path: string = '/data/activations.json'): Promise<ActivationData> {
	const response = await fetch(path);

	if (!response.ok) {
		throw new Error(`Failed to load activation data: ${response.statusText}`);
	}

	const data = await response.json();
	return data as ActivationData;
}

/**
 * Validate activation data structure
 */
export function validateActivationData(data: unknown): data is ActivationData {
	if (!data || typeof data !== 'object') return false;

	const d = data as Record<string, unknown>;

	// Check required top-level fields
	if (!d.metadata || typeof d.metadata !== 'object') return false;
	if (!d.vocabulary || typeof d.vocabulary !== 'object') return false;
	if (!d.specialists || !Array.isArray(d.specialists)) return false;
	if (!d.sequences || !Array.isArray(d.sequences)) return false;

	// Check metadata fields
	const meta = d.metadata as Record<string, unknown>;
	if (typeof meta.n_layers !== 'number') return false;
	if (typeof meta.n_neurons !== 'number') return false;

	return true;
}

/**
 * Get a summary of the activation data
 */
export function getDataSummary(data: ActivationData): {
	numSequences: number;
	numLayers: number;
	numNeurons: number;
	numSpecialists: number;
	vocabSize: number;
} {
	return {
		numSequences: data.sequences.length,
		numLayers: data.metadata.n_layers,
		numNeurons: data.metadata.n_neurons,
		numSpecialists: data.specialists.length,
		vocabSize: data.metadata.vocab_size
	};
}
