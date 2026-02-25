/**
 * Easing functions for smooth animations
 */

export function easeInOutCubic(t: number): number {
	return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function easeOutQuad(t: number): number {
	return 1 - (1 - t) * (1 - t);
}

export function easeInOutQuad(t: number): number {
	return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

/**
 * Linear interpolation between two values
 */
export function lerp(a: number, b: number, t: number): number {
	return a + (b - a) * t;
}

/**
 * Interpolate between two activation maps with easing
 */
export function interpolateActivations(
	fromActivations: Map<string, number>,
	toActivations: Map<string, number>,
	progress: number,
	easingFn: (t: number) => number = easeInOutCubic
): Map<string, number> {
	const result = new Map<string, number>();
	const easedProgress = easingFn(progress);

	// Interpolate values that exist in 'from'
	for (const [nodeId, fromValue] of fromActivations) {
		const toValue = toActivations.get(nodeId) ?? 0;
		result.set(nodeId, lerp(fromValue, toValue, easedProgress));
	}

	// Add values that only exist in 'to'
	for (const [nodeId, toValue] of toActivations) {
		if (!fromActivations.has(nodeId)) {
			result.set(nodeId, lerp(0, toValue, easedProgress));
		}
	}

	return result;
}

/**
 * Create a smooth transition controller for animations
 */
export function createTransitionController(durationMs: number = 300) {
	let startTime: number | null = null;
	let isAnimating = false;
	let frameId: number | null = null;

	return {
		start(onProgress: (progress: number) => void, onComplete?: () => void) {
			isAnimating = true;
			startTime = performance.now();

			const tick = (currentTime: number) => {
				if (!isAnimating || startTime === null) return;

				const elapsed = currentTime - startTime;
				const progress = Math.min(elapsed / durationMs, 1);

				onProgress(progress);

				if (progress < 1) {
					frameId = requestAnimationFrame(tick);
				} else {
					isAnimating = false;
					onComplete?.();
				}
			};

			frameId = requestAnimationFrame(tick);
		},

		stop() {
			isAnimating = false;
			if (frameId !== null) {
				cancelAnimationFrame(frameId);
				frameId = null;
			}
		},

		get isRunning() {
			return isAnimating;
		}
	};
}
