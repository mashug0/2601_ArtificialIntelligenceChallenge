// BDH Visualization Type Definitions

export interface ModelMetadata {
	model_name: string;
	n_layers: number;
	n_neurons: number;
	d_model: number;
	vocab_size: number;
	threshold: number;
}

export interface Vocabulary {
	char_to_int: Record<string, number>;
	int_to_char: Record<string, string>;
}

export interface SpecialistNeuron {
	neuron_idx: number;
	layer: number;
	trigger_char: string;
	activation_strength: number;
	selectivity: number;
}

export interface LayerActivation {
	layer_idx: number;
	sparsity: number;
	sampled_indices: number[];
	excitatory: number[][]; // [seq_len, sampled_neurons] - x_N values
	inhibitory: number[][]; // [seq_len, sampled_neurons] - y values
}

export interface Sequence {
	id: string;
	input_text: string;
	tokens: number[];
	layer_activations: LayerActivation[];
}

export interface ActivationData {
	metadata: ModelMetadata;
	vocabulary: Vocabulary;
	specialists: SpecialistNeuron[];
	sequences: Sequence[];
}

// Graph visualization types
export type SignalType = 'excitatory' | 'inhibitory' | 'inactive';

export interface GraphNode {
	id: string;
	layer: number;
	neuron_idx: number;
	is_specialist: boolean;
	specialist_char?: string;
	x?: number;
	y?: number;
	z?: number;
	vx?: number;
	vy?: number;
	vz?: number;
	// Dynamic properties (change per token)
	excitatory_activation: number;
	inhibitory_activation: number;
	signal_type: SignalType;
}

export interface GraphLink {
	source: string | GraphNode;
	target: string | GraphNode;
	weight: number;
	type: 'inter_layer' | 'intra_layer';
}

export interface GraphData {
	nodes: GraphNode[];
	links: GraphLink[];
}

// Store types
export interface PlaybackState {
	currentToken: number;
	maxToken: number;
	speed: number;
	isPlaying: boolean;
	loop: boolean;
}

export interface SelectionState {
	hoveredNode: GraphNode | null;
	selectedNode: GraphNode | null;
}

export interface LayerState {
	expanded: boolean[];
	visible: boolean[];
}

// Color configuration
export interface ColorConfig {
	excitatory: string;
	inhibitory: string;
	inactive: string;
	specialist: string;
	edge: string;
}

// --- Comparison Page Types ---

export interface ComparisonMetadata {
	model_name_champion: string;
	model_name_challenger: string;
	n_layers: number;
	n_embd: number;
	n_head: number;
	bdh_neurons_per_layer: number;
	top_k_fraction: number;
	vocab_size: number;
	sparsity_sentence: string;
}

export interface SparsityComparison {
	bdh: number[];
	transformer: number[];
}

export interface ConceptBattleResult {
	bdh_top_neuron: number;
	bdh_concept_strength: number;
	bdh_noise_activations: Record<string, number>;
	transformer_top_neuron: number;
	transformer_concept_strength: number;
	transformer_noise_activations: Record<string, number>;
}

export interface VisualSample {
	phrase: string;
	tokens_decoded: string[];
	n_sampled_neurons: number;
	bdh_grid: number[][];
	transformer_grid: number[][];
}

export interface ComparisonData {
	metadata: ComparisonMetadata;
	sparsity_comparison: SparsityComparison;
	concept_battle: Record<string, ConceptBattleResult>;
	visual_sample: VisualSample;
}

// --- Explainer / Battle Arena Types ---

export interface ExplainerMetadata {
	input_text: string;
	tokens: string[];
	n_tokens: number;
	n_layers: number;
	n_embd: number;
	n_head: number;
	bdh_neurons_per_layer: number;
	top_k_fraction: number;
}

export interface BDHActiveNode {
	id: string;
	layer: number;
	neuron: number;
	value: number;
}

export interface BDHActiveLink {
	source: string;
	target: string;
}

export interface BDHStep {
	token_idx: number;
	active_nodes: BDHActiveNode[];
	active_links: BDHActiveLink[];
}

export interface TransformerStep {
	layer: number;
	attention_matrix: number[][];
}

export interface ExplainerData {
	metadata: ExplainerMetadata;
	bdh_steps: BDHStep[];
	transformer_steps: TransformerStep[];
	narrative: string[];
}

// --- Battle Arena Data Types ---

export interface BattleMetadata {
	input_text: string;
	tokens: string[];
	n_tokens: number;
	n_layers: number;
	n_embd: number;
	n_head: number;
	top_k_fraction: number;
	bdh_neurons_per_layer: number;
}

export interface BattleGraphNode {
	id: string;
	layer: number;
	neuron: number;
	value: number;
}

export interface BattleGraphLink {
	source: string;
	target: string;
}

export interface BattleTokenGraph {
	token_idx: number;
	n_active: number;
	n_total: number;
	/** Dict of node_id -> normalized activation weight (0-1) for magma gradient */
	active_nodes: Record<string, number>;
	active_links: BattleGraphLink[];
}

export interface BattleData {
	metadata: BattleMetadata;
	transformer_load: number;
	bdh_load: number;
	energy_savings: number;
	transformer_layer_loads: number[];
	bdh_layer_loads: number[];
	transformer_layers: number[][][];
	bdh_graph: BattleTokenGraph[];
}
