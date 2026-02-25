import { writable, derived } from 'svelte/store';
import type { SelectionState, GraphNode, LayerState } from '$lib/types';

// Node selection store
function createSelectionStore() {
	const { subscribe, set, update } = writable<SelectionState>({
		hoveredNode: null,
		selectedNode: null
	});

	return {
		subscribe,

		setHovered(node: GraphNode | null) {
			update((s) => ({ ...s, hoveredNode: node }));
		},

		setSelected(node: GraphNode | null) {
			update((s) => ({ ...s, selectedNode: node }));
		},

		clearSelection() {
			set({ hoveredNode: null, selectedNode: null });
		},

		toggleSelected(node: GraphNode) {
			update((s) => ({
				...s,
				selectedNode: s.selectedNode?.id === node.id ? null : node
			}));
		}
	};
}

export const selectionStore = createSelectionStore();

// Layer expansion/visibility store
function createLayerStore(numLayers: number = 8) {
	const initialState: LayerState = {
		expanded: new Array(numLayers).fill(true),
		visible: new Array(numLayers).fill(true)
	};

	const { subscribe, set, update } = writable<LayerState>(initialState);

	return {
		subscribe,

		toggleExpanded(layerIdx: number) {
			update((s) => {
				const expanded = [...s.expanded];
				expanded[layerIdx] = !expanded[layerIdx];
				return { ...s, expanded };
			});
		},

		toggleVisible(layerIdx: number) {
			update((s) => {
				const visible = [...s.visible];
				visible[layerIdx] = !visible[layerIdx];
				return { ...s, visible };
			});
		},

		expandAll() {
			update((s) => ({
				...s,
				expanded: new Array(s.expanded.length).fill(true)
			}));
		},

		collapseAll() {
			update((s) => ({
				...s,
				expanded: new Array(s.expanded.length).fill(false)
			}));
		},

		showAll() {
			update((s) => ({
				...s,
				visible: new Array(s.visible.length).fill(true)
			}));
		},

		hideAll() {
			update((s) => ({
				...s,
				visible: new Array(s.visible.length).fill(false)
			}));
		},

		reset() {
			set(initialState);
		}
	};
}

export const layerStore = createLayerStore();

// Derived store: active node (either hovered or selected)
export const activeNode = derived(selectionStore, ($selection) => {
	return $selection.hoveredNode || $selection.selectedNode;
});

// Derived store: number of expanded layers
export const expandedLayerCount = derived(layerStore, ($layers) => {
	return $layers.expanded.filter(Boolean).length;
});
