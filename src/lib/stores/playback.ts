import { writable, derived } from 'svelte/store';
import type { PlaybackState } from '$lib/types';

function createPlaybackStore() {
	const { subscribe, set, update } = writable<PlaybackState>({
		currentToken: 0,
		maxToken: 0,
		speed: 1,
		isPlaying: false,
		loop: true
	});

	return {
		subscribe,

		setToken(idx: number) {
			update((s) => ({ ...s, currentToken: Math.max(0, Math.min(idx, s.maxToken)) }));
		},

		nextToken() {
			update((s) => {
				let nextIdx = s.currentToken + 1;
				if (nextIdx > s.maxToken) {
					if (s.loop) {
						nextIdx = 0;
					} else {
						return { ...s, isPlaying: false };
					}
				}
				return { ...s, currentToken: nextIdx };
			});
		},

		prevToken() {
			update((s) => ({
				...s,
				currentToken: Math.max(0, s.currentToken - 1)
			}));
		},

		setSpeed(speed: number) {
			update((s) => ({ ...s, speed }));
		},

		setMaxToken(max: number) {
			update((s) => ({ ...s, maxToken: max, currentToken: Math.min(s.currentToken, max) }));
		},

		play() {
			update((s) => ({ ...s, isPlaying: true }));
		},

		pause() {
			update((s) => ({ ...s, isPlaying: false }));
		},

		togglePlay() {
			update((s) => ({ ...s, isPlaying: !s.isPlaying }));
		},

		toggleLoop() {
			update((s) => ({ ...s, loop: !s.loop }));
		},

		reset() {
			update((s) => ({ ...s, currentToken: 0 }));
		},

		setPlayback(state: Partial<PlaybackState>) {
			update((s) => ({ ...s, ...state }));
		}
	};
}

export const playbackStore = createPlaybackStore();

// Derived store for progress percentage
export const progressPercent = derived(playbackStore, ($playback) =>
	$playback.maxToken > 0 ? ($playback.currentToken / $playback.maxToken) * 100 : 0
);
